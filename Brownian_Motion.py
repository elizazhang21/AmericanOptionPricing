# Brownian_Motion.py

import numpy as np
import scipy


def BM(S0, r, sigma, T, step):

    # generate geometric Brownian process:
    # dS = rSdt + sigma*dW
    # a pair of paths is generated each time, where one is antithetic

    S = np.zeros((2, T * step + 1))
    dt = 1 / step

    S[0][0] = S0
    S[1][0] = S0
    for i in range(1, T * step + 1):
        Z = np.random.randn()
        S[0][i] = S[0][i - 1] + S[0][i - 1] * (r * dt + sigma * dt ** 0.5 * Z)
        S[1][i] = S[1][i - 1] + S[1][i - 1] * (r * dt - sigma * dt ** 0.5 * Z)
    return S
