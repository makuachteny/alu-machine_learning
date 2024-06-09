#!/usr/bin/env python3

"""This module contains the function that
calculates the weighted moving average of a data set:
"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    data - is a list of data to calculate the moving average of
    beta - is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    m_avg = []
    v = 0
    for i in range(len(data)):
        v = (beta * v) + ((1 - beta) * data[i])
        m_avg.append(v / (1 - (beta ** (i + 1))))
    return m_avg
