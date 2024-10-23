#!/usr/bin/env python3
"""
Defines function that calculates that most likely sequence of hidden states for
the Hidden Markov Model
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states
    for the Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
            T: number of observations
        Emission [numpy.ndarray of shape (N, M)]:
            contains the emission probability of a specific observation
                given a hidden state
            N: number of hidden states
            M: number of all possible observations
        Transition [2D numpy.ndarray of shape (N, N)]:
            contains the transition probabilities
            Transition[i, j] is the probabilitiy of transitioning
                from the hidden state i to j
        Initial [numpy.ndarray of shape (N, 1)]:
            contains the probability of starting in a particular hidden state

    returns:
        path, P:
            path [list of length T]:
                contains the most likely sequence of hidden states
            P [float]:
                the probability of obtaining the path sequence
        or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    N1, N2 = Transition.shape
    if N1 != N or N2 != N:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    N3, N4 = Initial.shape
    if N3 != N or N4 != 1:
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    back = np.zeros((N, T))
    for i in range(1, T):
        F[:, i] = np.max(
            F[:, i - 1] * Transition.T * Emission[np.newaxis, :,
                                                  Observation[i]].T, axis=1)
        back[:, i] = np.argmax(
            F[:, i - 1] * Transition.T, axis=1)
    P = np.max(F[:, -1])
    Path = [np.argmax(F[:, -1])]
    for i in range(T - 1, 0, -1):
        Path.insert(0, int(back[Path[0], i]))
    return Path, P
