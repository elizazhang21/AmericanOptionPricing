# OptionValuer.py

import numpy as np
import scipy
from scipy.stats import norm
from scipy.linalg import inv
from scipy import stats
from sklearn.linear_model import LinearRegression


def Call_Payoff(stock_price, t, strike_price):
    return max(stock_price[t] - strike_price, 0)


def Put_Payoff(stock_price, t, strike_price):
    return max(strike_price - stock_price[t], 0)


def OptionValuer(paths, payoff, strike, r, times, steps):
    # paths is a N-step matrix of simulated asset price
    # payoff is a function compute exercise value at certain t and asset price
    # (early exercise) payoff of option
    # r is risk free rate used to compute discount factor
    # times is exercise times every year
    # steps are number of steps between exercise time
    # discount is discount factor between each steps

    N = len(paths)   # number of simulated paths (numbers of arrays)
    K = len(paths[0]) // steps   # number of total time steps (numbers of columns)
    discount = np.exp(-r / times)

    # declare variables for computing
    Y = np.zeros(N)
    P = np.zeros(N)
    X = np.zeros((N, 3))

    # initialize Y_t+1 with option value at maturity
    for i in range(N):
        Y[i] = payoff(paths[i], K * steps, strike)

    for i in range(K - 1, 1, -1):
        # prepare early exercise value vector
        for j in range(N):
            P[j] = payoff(paths[j], i * steps, strike)
        # P = np.apply_along_axis(payoff, 1, paths, i * steps, strike)

        # discount T_t+1
        Y = Y * discount

        # prepare lagurre polinomial values of asset price
        count = 0
        for j in range(N):
            if P[j] > 0:
                count = count + 1
                x = paths[j][i * steps]
                # print(x)
                X[j][0] = (1 - x)
                X[j][1] = (1 - 2 * x + 0.5 * x ** 2)
                X[j][2] = (1 - 3 * x + 1.5 * x ** 2 - x ** 3 / 6)

        if count >= 0.01 * N:
            X_train = []
            Y_train = []
            for j in range(N):
                if P[j] > 0:
                    X_train.append(X[j, :])
                    Y_train.append(Y[j])

            # solve lsm
            model = LinearRegression()
            model.fit(X_train, Y_train)

            predictions = model.predict(X)

            Y = np.maximum(P, np.where(P > 0, predictions, Y))
            # print('R-squared: ', model.score(X_test, Y_test))

        else:
            print('Too few paths')
            # too few paths have positive early exercise value, skip fit
            Y = np.maximum(P, Y)
    # print(Y)
    return Y
