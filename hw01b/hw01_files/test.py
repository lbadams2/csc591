import numpy as np
import matplotlib.pyplot as plt
import pandas

# Function that creates the X matrix as defined for fitting our model
# Each row is x_i (scalar) getting raised to increasing powers
def create_X(x,deg):
    X = np.ones((len(x),deg+1))
    for i in range(1,deg+1):
        X[:,i] = x**i # :, gets entire ith column
                      # x is entire x column from csv, it appears ** is applied to whole column at once
    return X

# Function for predicting the response
def predict_y(x,beta):
    created_x = create_X(x,len(beta)-1)
    return np.dot(created_x,beta)

# Function for fitting the model
def fit_beta(df,deg):
    return np.linalg.lstsq(create_X(df.x,deg),df.y,rcond=None)[0]

# Function for computing the MSE
def mse(y,yPred):
    return np.mean((y-yPred)**2)

# Loading training, validation and test data
dfTrain = pandas.read_csv('/Users/liam_adams/my_repos/csc591/hw01b/hw01_files/Data_Train.csv')
dfVal = pandas.read_csv('/Users/liam_adams/my_repos/csc591/hw01b/hw01_files/Data_Val.csv')
dfTest = pandas.read_csv('/Users/liam_adams/my_repos/csc591/hw01b/hw01_files/Data_Test.csv')

df_list = []
df_list.append(dfTrain)
df_list.append(dfVal)
df = pandas.concat(df_list, 0)
test = df.x

############ TRAINING A MODEL

# Fitting model
deg = 1
X = create_X(dfTrain.x,deg) # X is [len(dftrain.x), deg+1] matrix
beta = fit_beta(dfTrain,deg) # beta = (X^TX)^-1 X^TY which is deg + 1 column vector

# Computing training error
yPredTrain = predict_y(dfTrain.x,beta) # multiply each row in X by beta column vector to get prediction for each row
err = mse(dfTrain.y,yPredTrain) # calculate loss for each predction len(x) column vector
print('Training Error = {:2.3}'.format(err))

# Computing test error
yPredTest = predict_y(dfTest.x,beta)
err = mse(dfTest.y,yPredTest)
print('Test Error = {:2.3}'.format(err))