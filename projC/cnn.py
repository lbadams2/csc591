import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import soy_data

'''
Load 2D images, create filters (kernel matrices), apply each filter to every channel (r, g, b), and apply ReLU
Pool output of filters and pass to next layer and so on, flatten all pooled outputs into one long vector
and pass to fully connected layer. Then apply softmax, loss function, and backpropagation
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d - (in_channels, out_channels, kernel_size). Stride defaults to 1, padding defaults to 0
        self.conv1 = nn.Conv2d(3, 6, 5) # 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # this is second kernel matrix over the pooled values, 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # this is fully connected layer, 16*5*5 is 16 output channels and 5x5 kernel in each channel
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        test = self.conv1(x.float())
        test = F.relu(test)
        x = self.pool(test)
        #x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # flatten into vector before passing to fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainloader = soy_data.load_data()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs = data['image']
        labels = data['y']
        print('Image shape:', inputs.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')