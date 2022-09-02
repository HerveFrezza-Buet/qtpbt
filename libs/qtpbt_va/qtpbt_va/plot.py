import numpy as np
import matplotlib.pyplot as plt
from . import estimator

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def function_1d(ax, f, inf, sup, nb, vectorized = True, orientation = 'xy', color=colors[0]):
    Xs = np.linspace(inf, sup, nb, endpoint=True)
    if vectorized:
        Ys = f(Xs)
    else:
        Ys = np.array([f(x) for x in Xs])
    if orientation == 'xy':
        ax.plot(Xs, Ys, c=color)
    else:
        ax.plot(Ys, Xs, c=color)
    


def spline_1d(ax, f, inf, sup, nb, orientation = 'xy', color=colors[0]):
    function_1d(ax, f, inf, sup, nb, orientation)
    if orientation == 'xy':
        ax.scatter(f.Xs, f.Ys, c=color)
    else:
        ax.scatter(f.Ys, f.Xs, c=color)

def joint(ax_ij, ax_i, ax_j, i_name, j_name, samples,  bin_width=.05, degree=2, offset=.1,
          sample_color = colors[1], density_color = colors[0]):
    nb_samples = len(samples)
    ax_ij.set_aspect('equal')
    ax_ij.set_title(f'({i_name}, {j_name}), joint')
    ax_ij.set_xlim((0.0, 1.0))
    ax_ij.set_ylim((0.0, 1.0))
    ax_ij.scatter(samples[...,0], samples[..., 1], alpha = .1, color = sample_color)

    ax_j.set_aspect('equal')
    ax_j.set_title(f'{j_name}, marginalization')
    ax_j.set_xlim((0.0, 0.5))
    ax_j.set_ylim((0.0, 1.0))
    pJ  = estimator.density_1d(samples[..., 1], bin_width=.05, degree=2)
    ax_j.scatter(np.zeros(nb_samples) + offset, samples[..., 1], alpha = 0.01, color = sample_color)
    function_1d(ax_j, pJ, 0, 1, 200, orientation = 'yx', color = density_color)

    ax_i.set_aspect('equal')
    ax_i.set_title(f'{i_name}, marginalization')
    ax_i.set_xlim((0.0, 1.0))
    ax_i.set_ylim((0.0, 0.5))
    pI  = estimator.density_1d(samples[..., 0], bin_width=.05, degree=2)
    ax_i.scatter(samples[..., 0], np.zeros(nb_samples) + offset, alpha = 0.01, color = sample_color)
    function_1d(ax_i, pI, 0, 1, 200, orientation = 'xy', color = density_color)
    
