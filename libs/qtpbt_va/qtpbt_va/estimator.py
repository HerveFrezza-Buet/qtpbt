import numpy as np
import scipy.interpolate
import scipy.integrate
from . import utils

import matplotlib.pyplot as plt

def expectancy(sampler, nb_samples):
    """
    sampler : should provide scalar samples
    nb_samples : We use many tosses to estimate the expectancy.
    returns : the estimated expectancy.
    """
    return np.mean(sampler(nb_samples)['samples'])

def density_1d(samples, inf=None, sup=None, bin_width=.05, nb_bins=100):
    Xs = samples
    Y = []
    bw = bin_width * .5
    if inf == None:
        inf = np.min(Xs)
    if sup == None:
        sup = np.max(Xs)
    X = np.linspace(inf, sup, nb_bins, endpoint=True)
    for x in X:
        inf, sup = x - bw, x + bw
        Bin = Xs[np.logical_and(Xs >= inf, Xs <= sup)]
        Y.append(len(Bin))
    Y = np.array(Y)
    integ = scipy.integrate.trapezoid(Y, x=X)
    Y = Y * (1.0 / integ)
    return utils.LinearInterpolator(X, Y)







