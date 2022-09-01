import numpy as np

def expectancy(sampler, nb_samples):
    """
    sampler : should provide scalar samples
    nb_samples : We use many tosses to estimate the expectancy.
    returns : the estimated expectancy.
    """
    return np.mean(samples(nb_samples)['samples'])

