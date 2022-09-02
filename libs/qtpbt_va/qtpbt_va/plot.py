import numpy as np

def function_1d(ax, f, inf, sup, nb, vectorized = True, orientation = 'xy'):
    Xs = np.linspace(inf, sup, nb, endpoint=True)
    if vectorized:
        Ys = f(Xs)
    else:
        Ys = np.array([f(x) for x in Xs])
    if orientation == 'xy':
        ax.plot(Xs, Ys)
    else:
        ax.plot(Ys, Xs)
    


def spline_1d(ax, f, inf, sup, nb, orientation = 'xy'):
    function_1d(ax, f, inf, sup, nb, orientation)
    if orientation == 'xy':
        ax.scatter(f.Xs, f.Ys, c='k')
    else:
        ax.scatter(f.Ys, f.Xs, c='k')
    
