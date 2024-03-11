#!/usr/bin/env python3
'''This module calculates the likelihood of obtaining data'''
import numpy as np


def likelihood(x, n, P):
  """
  Calculates the likelihood of obtaining data (x patients with severe side effects out of n)
  given various hypothetical probabilities (P) of developing severe side effects.

  Args:
      x: Number of patients with severe side effects (int, non-negative, <= n)
      n: Total number of patients observed (positive integer)
      P: 1D numpy array containing hypothetical probabilities (0 to 1)

  Returns:
      A 1D numpy array containing the likelihood for each probability in P.

  Raises:
      ValueError: If n is not a positive integer, x is not a valid integer,
                 x is greater than n, P is not a 1D numpy array, or values in P
                 are not in the range [0, 1].
  """

  # Input validation
  if not isinstance(n, int) or n <= 0:
    raise ValueError("n must be a positive integer")
  if not isinstance(x, int) or x < 0:
    raise ValueError("x must be an integer that is greater than or equal to 0")
  if x > n:
    raise ValueError("x cannot be greater than n")
  if not isinstance(P, np.ndarray) or P.ndim != 1:
    raise TypeError("P must be a 1D numpy.ndarray")
  if np.any(P < 0) or np.any(P > 1):
    raise ValueError("All values in P must be in the range [0, 1]")

  # Calculate likelihood using binomial probability formula
  likelihoods = np.where(x == 0, (1 - P)**n,
                         binom(n, x) * P**x * (1 - P)**(n-x))
  return likelihoods