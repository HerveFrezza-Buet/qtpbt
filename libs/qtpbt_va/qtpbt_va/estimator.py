import numpy as np
import scipy.interpolate
from . import utils

def expectancy(sampler, nb_samples):
    """
    sampler : should provide scalar samples
    nb_samples : We use many tosses to estimate the expectancy.
    returns : the estimated expectancy.
    """
    return np.mean(sampler(nb_samples)['samples'])

def density_1d(samples, bin_width=.05):
    Xs = np.sort(samples)
    nb_points = int((Xs[-1] - Xs[0]) / bin_width + .5)
    X = np.linspace(Xs[0], Xs[-1], nb_points, endpoint=True)
    Y = []
    bw = bin_width * .5
    for x in X:
        inf, sup = x - bw, x + bw
        Bin = Xs[np.logical_and(Xs >= inf, Xs <= sup)]
        Y.append(len(Bin))
    Y = np.array(Y)
    Y = Y * (1.0 / np.sum(Y))
    return utils.Spline(X, Y, 3)

