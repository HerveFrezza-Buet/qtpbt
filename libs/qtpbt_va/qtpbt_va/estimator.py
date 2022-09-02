import numpy as np
import scipy.interpolate

def expectancy(sampler, nb_samples):
    """
    sampler : should provide scalar samples
    nb_samples : We use many tosses to estimate the expectancy.
    returns : the estimated expectancy.
    """
    return np.mean(sampler(nb_samples)['samples'])

def density_1d(samples, bin_width=.05):
    Xs = np.sort(samples)
    Ys = []
    bw = bin_width * .5
    for x in Xs:
        inf, sup = x - bw, x + bw
        Bin = Xs[np.logical_and(Xs >= inf, Xs <= sup)]
        Ys.append(len(Bin))
    Ys = np.array(Ys)
    integral_ = 1 / np.sum(Ys)
    return scipy.interpolate.UnivariateSpline(Xs, Ys * integral_, k=3)

