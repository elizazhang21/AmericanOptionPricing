# RandomNumberGenerator.py
import numpy as np


def Marsaglia_Bray_Generator(N):
    x = []
    for i in range(N):
        w = 2
        while(w > 1.0):
            x1 = 2 * np.random.ranf() - 1
            x2 = 2 * np.random.ranf() - 1
            w = x1 * x1 + x2 * x2
        w = (-2 * np.log(w) / w) ** 0.5
        y1 = x1 * w
        y2 = x2 * w
        x.append(y1)
        x.append(y2)
        # return standard_gaussian_random_number
    return x
