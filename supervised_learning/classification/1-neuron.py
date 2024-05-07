#!/usr/bin/env python3
"""Neuron class that defines a single neuron performing binary classification. (Based on 0-neuron.py)"""

import numpy as np

class Neuron:
    """ Neuron class that defines a single neuron performing binary classification """
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def get_W(self):
        """ Getter method for W(Weight) """
        return self.__W

    @property
    def get_b(self):
        """ Getter method for b(Bias) """
        return self.__b

    @property
    def get_A(self):
        """ Getter method for A(Activation) """
        return self.__A
