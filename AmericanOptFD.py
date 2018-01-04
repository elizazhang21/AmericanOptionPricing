# AmericanOptFD.py
import numpy as np
import scipy as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt


def Call(stock_price, strike_price):
    return max(stock_price - strike_price, 0)


def Put(stock_price, strike_price):
    return max(strike_price - stock_price, 0)


def AmeOptFD(S0, K, r, T, sigma, N, M, type):
    # price an American option via finite differences
    # return the price of an American option computed using finite difference
    # method

    # create time grid
    t = np.linspace(0, T, N + 1)
    dt = T / N   # Time step
    delta = 0

    # share price grid
    s_max = 2 * max(S0, K) * np.exp(r * T)
    s_min = 0
    ds = s_max / M
    s = np.linspace(0, s_max, M + 1)
    # check the stability condition
    if 0 < dt / ds ** 2 < 0.5:
        pass
    else:
        return None

    # Now find points either side of the initial price so that we can calculate
    # the price of the option via interpolation
    idx = []
    count = -1
    for si in s:
        count += 1
        if si < S0:
            idx.append(count)
    idx = idx[-1]

    a = S0 - s[idx]
    b = s[idx + 1] - S0
    Z = 1 / (a + b) * np.matrix([a, b])

    # Set up a pricing matrix to hold the values we compute
    P = np.zeros((N + 1, M + 1))

    # Boundary condition
    if type == Put:
        # Value of option at maturity - Put
        P[-1, :] = np.maximum(K - np.linspace(0, M, M + 1) * ds, 0)
    else:
        # Value of option at maturity - Call
        P[-1, :] = np.maximum((np.linspace(0, M, M + 1) * ds - K, 0))
    P[:, 0] = K
    # Value of option when stock price is 0
    P[:, -1] = 0
    # Value of option when S = Smax

    # form the weighting matrix A
    A = np.zeros((M - 1, M - 1))
    a = np.zeros(M - 1)
    b = np.zeros(M - 1)
    c = np.zeros(M - 1)
    for i in range(1, M):
        a[i - 1] = r / 2 * i * dt - 0.5 * sigma ** 2 * i ** 2 * dt
        b[i - 1] = 1 + sigma ** 2 * i ** 2 * dt + r * dt
        c[i - 1] = -r / 2 * i * dt - 0.5 * sigma ** 2 * i ** 2 * dt
    a = a[1:]
    c = c[:-1]
    diagonals = [b, a, c]
    A = diags(diagonals, [0, -1, 1]).todense()

    for i in range(N, 0, -1):
        addition = np.zeros(M - 1).T
        addition[0] = -(r / 2 * dt - 0.5 * sigma ** 2 * dt) * K
        addition[-1] = 0
        y = P[i, 1:-1].T + addition
        x = np.linalg.inv(A).dot(y)

        if type == Put:
            P[i - 1, 1:-1] = np.maximum(x, K - s[1:-1])

        else:
            for j in range(1, N - 1):
                P[i, j] = max(x, s[j] - k)

    P_FD = Z.dot(P[0, idx - 1:idx + 1])
    return P_FD
