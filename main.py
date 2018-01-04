# main.py

import numpy as np
import scipy
import functools as ft
from functools import reduce
from OptionValuer import OptionValuer
from Brownian_Motion import BM
import matplotlib.pyplot as plt
from JumpDiffusion import *
from AmericanOptFD import *
# from binomial import *

# option parameters:
S_seq = np.linspace(34, 46, 13, endpoint=True)
r = 0.06
sigma = 0.2
T = 1
strike = 40

# payoff function


def Call_Payoff(stock_price, t, strike_price):
    return max(stock_price[t] - strike_price, 0)


def Put_Payoff(stock_price, t, strike_price):
    return max(strike_price - stock_price[t], 0)


# ========================================================================================
# =========================        American Option Valuer        =========================
# ========================================================================================

# number of exercise points in one year
times = 50
# number of simulations, total paths number would be 2N
N = 1000
# steps between exercise points when generating paths
step = 50
step_year = step * times

paths = np.zeros((2 * N, step_year * T + 1))
P = np.zeros(len(S_seq))
FD = np.zeros(len(S_seq))

for k in range(len(S_seq)):
    S = S_seq[k]
    for i in range(N):
        bm = BM(S, r, sigma, T, step_year)
        paths[2 * i] = bm[0]
        paths[2 * i + 1] = bm[1]
    rr = OptionValuer(paths, Put_Payoff, strike, r, times, step)
    P[k] = np.mean(rr) * np.exp(-r / times)
    FD[k] = AmeOptFD(S, strike, r, 1, sigma, 2500, 1000, Put)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(S_seq, P, 'k-', c='blue', label='Simulation')
plt.scatter(S_seq, FD, c='red', label='FD Method')
legend = ax.legend(loc='upper right', fontsize=8)
plt.xlabel('Initial Prices', fontsize=8)
plt.ylabel('Simulated Option Prices', fontsize=8)
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(6)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(6)
plt.title('Valuing American Put Option', fontsize=8)
plt.grid()
plt.show()

# =================================================================================
# =========================        Binomial Method        =========================
# =================================================================================


def BPTree(n, S, u, d):
    r = [np.array([S])]
    for i in range(n):
        r.append(np.concatenate((r[-1][:1] * u, r[-1] * d)))
    return r


def GBM(R, P, S, T, r, b, v, n):
    t = float(T) / n
    u = np.exp(v * np.sqrt(t))
    d = 1. / u
    p = (np.exp(b * t) - d) / (u - d)
    ptree = BPTree(n, S, u, d)[::-1]
    R_ = ft.partial(R, np.exp(-r * t), p)
    return reduce(R_, map(P, ptree))[0]


def American(D, p, a, b):
    return np.maximum(b, D * (a[:-1] * p + a[1:] * (1 - p)))


def Call(S, K):
    return np.maximum(S - K, 0)


def Put(S, K):
    return np.maximum(K - S, 0)

ABM = ft.partial(GBM, American)

price_call = []
price_put = []

S0 = np.linspace(34, 46, 13, endpoint=True)
for i in range(len(S0)):
    price_call.append(ABM(ft.partial(Call, K=40.0), S0[i], 1.0, 0.06, 0.06, 0.2, 1000))
    price_put.append(ABM(ft.partial(Put, K=40.0), S0[i], 1.0, 0.06, 0.06, 0.2, 1000))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(S0, price_call, 'k-', c='blue', label='Call Option Prices')
ax.plot(S0, price_put, 'k-', c='red', label='Put Option Prices')
legend = ax.legend(loc='upper right', fontsize=8)
plt.xlabel('Initial Prices', fontsize=8)
plt.ylabel('Simulated Option Prices', fontsize=8)
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(6)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(6)
plt.title('Valuing American Call Option', fontsize=8)
plt.grid()
plt.show()
