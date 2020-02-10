#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.append('/Users/liam_adams/my_repos/csc591/hw02b/assignment')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '/Users/liam_adams/my_repos/csc591/hw02b/data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def test():
    train_data, valid_data, test_data = load_data()
    model = network2.load('model.json')
    correct, results = model.accuracy(test_data)
    acc = correct / len(test_data[0])
    print('Test accuracy:', str(acc))

    predictions = [network2.vectorized_result(x) for (x, y) in results]
    with open('predictions.csv', 'w') as f:
        wr = csv.writer(f)
        for vec in predictions:
            li = vec.flatten().tolist()
            wr.writerow(li)


def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 200, 10])
    # train the network using SGD
    eval_cost, eval_acc, train_cost, train_acc = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 2,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    plt.plot(eval_cost)
    plt.plot(train_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Validation', 'Training'], loc='upper left')
    plt.show()

    tr_acc_per = [x/len(train_data[0]) for x in train_acc]
    val_acc_per = [x/len(valid_data[0]) for x in eval_acc]
    plt.plot(val_acc_per)
    plt.plot(tr_acc_per)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['Validation', 'Training'], loc='upper left')
    plt.show()
    model.save('model.json')

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test()
