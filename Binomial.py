import functools as ft
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt


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


def VP(S, K):
    return np.maximum(S - K, 0)
ABM = ft.partial(GBM, American)

price = []

S0 = np.linspace(34, 46, 13, endpoint=True)
for i in range(len(S0)):
    price.append(ABM(ft.partial(VP, K=40.0), S0[i], 1.0, 0.06, 0.06, 0.2, 1000))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(S0, price, 'k-', c='blue')
# legend = ax.legend(loc='upper right', fontsize=8)
plt.xlabel('Initial Prices', fontsize=8)
plt.ylabel('Simulated Option Prices', fontsize=8)
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(6)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(6)
plt.title('Valuing American Call Option', fontsize=8)
plt.grid()
plt.show()
