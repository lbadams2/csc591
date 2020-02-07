#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    ans = 1/(1 + np.exp(-z))
    return ans

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    ans = sigmoid(z)*(1 - sigmoid(z))
    return ans
