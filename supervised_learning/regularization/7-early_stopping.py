#!/usr/bin/env python3

"""this module has a function that determines
if you should stop gradient descent early"""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    cost - current validation cost
    opt_cost - lowest recorded validation cost
    threshold - threshold for early stopping
    patience - count used for early stopping
    count - count of how long threshold has not been met
    """

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count

    return False, count
