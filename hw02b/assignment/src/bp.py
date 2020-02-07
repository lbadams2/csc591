#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y -> x is a single 784 x 1 np array, y is a 10 x 1 np array
            biases, weights (list): list of biases and transposed weights of entire network
            
            biases is list of 2 vectors for final 2 (of 3) layers. First is 20 x 1, second is 10 x 1
            
            weightsT is list of 2 matrices. 20x784 weight matrix and 10x20 weight matrix. The first matrix
            transforms the 784 dim input to 20 dim, then the 20 dim to 10 dim.  It operates on one row of
            x at a time

            cost (CrossEntropyCost): object of cost computation - binary cross entropy on each of the 10 outputs
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]
    n = len(x)
    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    activations = []
    x_input = x
    for b, wT in zip(biases, weightsT):
        #layer_activations = np.zeros(b.shape)
        #layer_activations = [[0] * b.shape[0]] * n # activation for each of the n inputs will be a vector with same dim as bias (size of hidden layer)
        #test = np.dot(wT, x)
        #test = test + b
        #test2 = sigmoid(test)
        x = sigmoid(np.dot(wT, x)+b) # xi is vector with same size as bias (size of hidden layer)            
        #layer_activations[i] = x # set row to activation vector of one input for the current hidden layer
        activations.append(x)


    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y) # delta is 10 x 1, same dims as last ouput layer (y-hat) and y
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    # delta is [n x 10] matrix of gradients of loss function evaluated at each y_hat
    for k in range(num_layers - 2, -1, -1):
        a = activations[k] # activation vector for lth layer - 10 x 1 for last layer
        a_prime = sigmoid_prime(a) # this is still 10 x 1 or h x 1
        delta = delta * a_prime # this should give element wise multiplication of the matrices - 10 x 1 or h x 1
        nabla_b[k] = delta # if no lambda and regularization
        if k > 0:
            weight = activations[k-1].dot(delta.T) # this will be [20 x 1] dot [1 x 10] = 20 x 10
            nabla_wT[k] =  weight.T
        else:
            weight = x_input.dot(delta.T)
            nabla_wT[k] = weight.T
        w_untranspose = weightsT[k].T
        delta = w_untranspose.dot(delta)

    return (nabla_b, nabla_wT)

