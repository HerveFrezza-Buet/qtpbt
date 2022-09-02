import numpy as np
import scipy.interpolate

class Spline:
    """
    Xs, Ys are the coordinates of the points to interpolate.
    !!! Xs must be unique values.
    """
    def __init__(self, Xs, Ys, degree=3):
        self.degree = degree
        self.Xs = Xs
        self.Ys = Ys
        splines = scipy.interpolate.splrep(Xs, Ys, k=degree)
        self.f = lambda X, splines=splines : scipy.interpolate.splev(X, splines)

    def __call__(self, Xs):
        return self.f(Xs)
