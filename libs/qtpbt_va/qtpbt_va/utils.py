import numpy as np
import scipy.interpolate
import sklearn.svm

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

class SVR:
    def __init__(self, Xs, Ys, epsilon=.01, C=2, sigma=.01):
        self.Xs = Xs
        self.Ys = Ys
        reg = sklearn.svm.SVR(C=C, kernel='rbf', gamma = .5/(sigma*sigma), tol=1e-8)
        reg.fit(Xs.reshape(-1,1), Ys)
        self.f = lambda X, reg = reg : reg.predict(X.reshape(-1,1))

    def __call__(self, Xs):
        return self.f(Xs)
    

class LinearInterpolator:
    def __init__(self, Xs, Ys):
        args = np.argsort(Xs)
        Xs = Xs[args]
        Ys = Ys[args]
        self.Xs = np.append(np.insert(Xs, 0, Xs[0]), Xs[-1])
        self.Ys = np.append(np.insert(Ys, 0, 0), 0)
        self.f = scipy.interpolate.interp1d(Xs, Ys)

        
    def __call__(self, Xs):
        return self.f(Xs)
    
