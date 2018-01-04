# MonteCarloForStockPrice.py
import numpy as np
import matplotlib.pyplot as plt
from RandomNumberGenerator import Marsaglia_Bray_Generator


def Jump_Diffusion_Process_At_Fixed_Dates(S, maturity, n, mu, sigma, d, Lambda, a, b):
    # Simulate Jump-Diffusion Process using Algorithm 5.6
    # mu and sigma are paramethers of normal GBM, lambda is parameter of poisson distribution
    # a and b are patameters of random variable Y

    time = np.linspace(0, maturity, n)
    dt = maturity / (n - 1)
    X = np.log(S)
    PriceList = [S]
    Z = Marsaglia_Bray_Generator(n - 1)
    for i in range(n - 1):
        # 1.generate Z ~ N (0,1)
        Z1 = Z[2 * i]
        Z2 = Z[2 * i + 1]
        # 2.generate N ~ P(lambda(t_i+1 - t_i))
        N = np.random.poisson(Lambda * dt)
        if N <= 0:
            M = 0
            # skip to step 4
        else:
            M = a * N + b * N ** 0.5 * Z2
        X = X + (mu - d - 0.5 * sigma ** 2) * dt + sigma * dt ** 0.5 * Z1 + M
        S = np.exp(X)
        PriceList.append(S)

    return PriceList
