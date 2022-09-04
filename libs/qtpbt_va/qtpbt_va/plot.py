import numpy as np
import matplotlib.pyplot as plt
from . import estimator

plt.rcParams['text.usetex'] = True
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def function_1d(ax, f, inf, sup, nb, vectorized = True, orientation = 'xy', color=colors[0], alpha=1.0, label=None):
    Xs = np.linspace(inf, sup, nb, endpoint=True)
    if vectorized:
        Ys = f(Xs)
    else:
        Ys = np.array([f(x) for x in Xs])
    if orientation == 'xy':
        ax.plot(Xs, Ys, c=color, alpha=alpha, label=label)
    else:
        ax.plot(Ys, Xs, c=color, alpha=alpha, label=label)
    


def spline_1d(ax, f, inf, sup, nb, orientation = 'xy', color=colors[0]):
    function_1d(ax, f, inf, sup, nb, orientation)
    if orientation == 'xy':
        ax.scatter(f.Xs, f.Ys, c=color)
    else:
        ax.scatter(f.Ys, f.Xs, c=color)

def proba_1d(ax, fx_sampler, inf, sup, nb_samples, test, var_name, test_name, alpha=.01):
    data = fx_sampler(nb_samples)
    function_1d(ax, fx_sampler.f, 0, 1, 200)
    ax.set_ylabel(r'${}$'.format(var_name))
    data = np.stack((data['samples'], data['from_samples'])).T
    data_pass = []
    data_fail = []
    for d in data:
        if test(d[0]) :
            data_pass.append(d)
        else:
            data_fail.append(d)
    data_pass = np.array(data_pass)
    data_fail = np.array(data_fail)
    proba = len(data_pass)/float(len(data))
    ax.set_xlabel(r'${}mathrm P {}left( {} {}right) = {:.2f}$'.format('\\', '\\', test_name, '\\', proba))
                                        
    plt.scatter(data_fail[..., 1], np.zeros(len(data_fail)), color=colors[0], alpha = alpha)    
    plt.scatter(np.zeros(len(data_fail)), data_fail[..., 0], color=colors[1], alpha = alpha)  
    plt.scatter(data_pass[..., 1], np.zeros(len(data_pass)), color=colors[2], alpha = alpha)    
    plt.scatter(np.zeros(len(data_pass)), data_pass[..., 0], color=colors[3], alpha = alpha)    
    plt.scatter(data_pass[..., 1], data_pass[..., 0], color='k', alpha = 2*alpha)  
    

def joint(ax_ij, ax_i, ax_j, i_name, j_name, samples, inf = None, sup = None,  bin_width=.05, offset=.1,
          sample_color = colors[1], density_color = colors[0], cond_thickness = None, i_cond = None, j_cond = None):
    nb_samples = len(samples)
    ax_ij.set_aspect('equal')
    ax_ij.set_title(r'$({}, {})$, joint'.format(i_name, j_name))
    ax_ij.set_xlim((0.0, 1.0))
    ax_ij.set_ylim((0.0, 1.0))
    ax_ij.scatter(samples[...,0], samples[..., 1], alpha = .1, color = sample_color)


    ax_i.set_title(f'{i_name}, marginalization')
    ax_i.set_xlim((0.0, 1.0))
    pI  = estimator.density_1d(samples[..., 0], bin_width=bin_width, inf=inf, sup=sup)
    ax_i.scatter(samples[..., 0], np.zeros(nb_samples) + offset, alpha = 0.01, color = sample_color)
    function_1d(ax_i, pI, 0, 1, 200, orientation = 'xy', color = density_color, label=r'${}mathrm p_{}{}{}$'.format('\\', '{', i_name, '}'))
    
    ax_j.set_title(f'{j_name}, marginalization')
    ax_j.set_ylim((0.0, 1.0))
    pJ  = estimator.density_1d(samples[..., 1], bin_width=bin_width, inf=inf, sup=sup)
    ax_j.scatter(np.zeros(nb_samples) + offset, samples[..., 1], alpha = 0.01, color = sample_color)
    function_1d(ax_j, pJ, 0, 1, 200, orientation = 'yx', color = density_color, label=r'${}mathrm p_{}{}{}$'.format('\\', '{', j_name, '}'))


    if i_cond:
        for idx, i_val in enumerate(i_cond):
            cinf, csup = i_val - cond_thickness*.5, i_val + cond_thickness*.5
            color = colors[idx+2]
            ax_ij.axvline(cinf, color = color, alpha=.3)
            ax_ij.axvline(csup, color = color, alpha=.3)
            cond_test = samples[..., 0]
            cond_data = samples[..., 1]
            cond_data = cond_data[np.logical_and(cond_test >= cinf, cond_test <= csup)]
            p = estimator.density_1d(cond_data, bin_width=bin_width, inf=inf, sup=sup)
            s = '\\left. {} \\middle| {} = {} \\right.'.format(j_name, i_name, i_val)
            function_1d(ax_j, p, 0, 1, 200, orientation = 'yx', color = color, alpha=.3, label=r'${}mathrm p_{}{}{}$'.format('\\', '{', s, '}'))

    if j_cond:
        for idx, j_val in enumerate(j_cond):
            cinf, csup = j_val - cond_thickness*.5, j_val + cond_thickness*.5
            color = colors[idx+2]
            ax_ij.axhline(cinf, color = color, alpha=.3)
            ax_ij.axhline(csup, color = color, alpha=.3)
            cond_test = samples[..., 1]
            cond_data = samples[..., 0]
            cond_data = cond_data[np.logical_and(cond_test >= cinf, cond_test <= csup)]
            p = estimator.density_1d(cond_data, bin_width=bin_width, inf=inf, sup=sup)
            s = '\\left. {} \\middle| {} = {} \\right.'.format(i_name, j_name, j_val)
            function_1d(ax_i, p, 0, 1, 200, orientation = 'xy', color = color, alpha=.3, label=r'${}mathrm p_{}{}{}$'.format('\\', '{', s, '}'))

    ax_i.legend()
    ax_j.legend()
            
