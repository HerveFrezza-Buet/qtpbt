import numpy as np

def function_1d(ax, f, inf, sup, nb, vectorized = True):
    Xs = np.linspace(inf, sup, nb, endpoint=True)
    if vectorized:
        Ys = f(Xs)
    else:
        Ys = np.array([f(x) for x in Xs])
    ax.plot(Xs, Ys)
    


def spline_1d(ax, f, inf, sup, nb):
    function_1d(ax, f, inf, sup, nb)
    ax.scatter(f.Xs, f.Ys, c='k')
    
