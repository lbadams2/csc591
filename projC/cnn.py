import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import soy_data
import torch
import matplotlib.pyplot as plt

'''
Load 2D images, create filters (kernel matrices), apply each filter to every channel (r, g, b), and apply ReLU
Pool output of filters and pass to next layer and so on, flatten all pooled outputs into one long vector
and pass to fully connected layer. Then apply softmax, loss function, and backpropagation
'''

####### Hyper Parameters ###############
batch_size = 16
learning_rate = .001
epochs = 10
kernel_size = 10
pool_kernel_size = 6
out_channel_1 = 6
out_channel_2 = 16
stride = 2
padding = 0
########################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d - (in_channels, out_channels, kernel_size). Stride defaults to 1, padding defaults to 0
        self.conv1 = nn.Conv2d(3, out_channel_1, kernel_size=kernel_size, stride=stride) # 10x10 kernel
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(out_channel_1, out_channel_2, kernel_size=kernel_size) # this is second kernel matrix over the pooled values, 10x10
        self.fc1 = nn.Linear(out_channel_2 * 51 * 71, 120) # this is fully connected layer, 51 and 71 come from ouput formula below
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 5)

# output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    def forward(self, x): # x is (4, 3, 480, 640)
        test = self.conv1(x.float()) # apply output formula above to last 2 dims (4, 6, 236, 316)
        test = F.relu(test) # this doesn't change dims
        x = self.pool(test) # apply output formula above to last 2 dims (4, 6, 116, 156)
        #x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # apply output formula and output channels (4, 16, 51, 71)
        x = x.view(-1, out_channel_2 * 51 * 71) # flatten into vector before passing to fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # could add softmax
        return x


def accuracy(predictions, labels):
    # need to get max value from each dim-5 prediction vector
    predictions = [torch.max(pred_vec, 0)[1] for pred_vec in predictions]
    #predictions = p for (_, p) in torch.max(predictions, 1)
    correct = sum(int(p == y) for (p, y) in zip(predictions, labels))
    return correct


def train(criterion, optimizer, batch_size, net):
    train_loader, val_loader, train_size, val_size = soy_data.load_data(batch_size)
    training_cost, training_accuracy = [], []
    evaluation_cost, evaluation_accuracy = [], []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total_train_loss = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs = data['image']
            labels = data['y']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            correct += accuracy(outputs, labels)
            # outputs will be (batch_size x 5) and labels will be (batch_size)
            # i think loss function still works with these mismatched dimensions
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_train_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        
        #print('number correct %d len training data %d' % (correct, train_size))
        epoch_train_acc = correct / train_size
        epoch_train_loss = total_train_loss / train_size
        training_accuracy.append(epoch_train_acc)
        training_cost.append(epoch_train_loss)
        print('Training accuracy epoch %d = %.2f' % (epoch, epoch_train_acc))
        print('Training loss epoch %d = %.2f' % (epoch, epoch_train_loss))

        total_val_loss = 0
        val_correct = 0
        for i, data in enumerate(val_loader, 0):
            
            #Wrap tensors in Variables
            inputs = data['image']
            labels = data['y']
            
            #Forward pass
            val_outputs = net(inputs)
            val_correct += accuracy(val_outputs, labels)
            val_loss = criterion(val_outputs, labels)
            total_val_loss += val_loss.item()
        
        epoch_val_acc = val_correct / val_size
        epoch_val_loss = total_val_loss / val_size
        evaluation_accuracy.append(epoch_val_acc)
        evaluation_cost.append(epoch_val_loss)
        print('Validation accuracy epoch %d  = %.2f' % (epoch, epoch_val_acc))
        print('Validation loss epoch %d = %.2f' % (epoch, epoch_val_loss))

    print('Finished Training')
    return training_accuracy, training_cost, evaluation_accuracy, evaluation_cost


def make_plots(train_acc, train_cost, eval_acc, eval_cost):
    plt.plot(eval_cost)
    plt.plot(train_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Validation', 'Training'], loc='upper left')
    plt.show()

    plt.plot(eval_acc)
    plt.plot(train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['Validation', 'Training'], loc='upper left')
    plt.show()


def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    ta, tc, ea, ec = train(criterion, optimizer, batch_size, net)
    make_plots(ta, tc, ea, ec)
    torch.save(net.state_dict(), 'model.mdl')


if __name__ == '__main__':
    main()