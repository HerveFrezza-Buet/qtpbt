
import sys
import numpy as np
import qtpbt_va as va
import matplotlib.pyplot as plt

Xs  = np.array([0.00, 0.01, 0.05, 0.10, 0.20, 0.50])
Ys  = np.array([0.00, 0.10, 0.30, 0.35, 0.40, 0.50])
XXs = 1.0 - np.flip(Xs[:-1])
YYs = 1.0 - np.flip(Ys[:-1])
Xs  = np.hstack((Xs, XXs))
Ys  = np.hstack((Ys, YYs))
h_sigmoid1 = va.utils.Spline(Xs, Ys, degree=3) 
h_sigmoid2 = va.utils.SVR(Xs, Ys, sigma=.3) 


def test_001():
    U = va.sampler.Uniform(-1, 1) # random values in [-1, 1]
    print(U(10))
    print('E(U) = {}'.format(va.estimator.expectancy(U, 1000)))

def test_002():
    U  = va.sampler.Uniform()
    UU = va.sampler.Nuplet(U, 2)
    print()
    print('UU')
    print()
    print(UU(10)['samples'])
    
    print()
    print()
    print('X')
    print()
    X = va.sampler.Nuplet(UU, 3)
    print(X(5)['samples'])

def test_003():
    U  = va.sampler.Uniform()
    f  = lambda x : 1/(1 + np.exp(-15*(x - .5)))
    fU = va.sampler.Apply(U, f, vectorized = True)
    fig = plt.figure(figsize = (10, 5))
    nb_samples = 300
    data = fU(nb_samples)
    va.plot.function_1d(fig.gca(), f, 0, 1, 200, vectorized = True)
    plt.scatter(data['from_samples'], np.zeros(nb_samples), alpha = .1)
    plt.scatter(np.zeros(nb_samples), data['samples'],      alpha = .1)
    plt.show()

def test_004():
    U  = va.sampler.Uniform()
    nb_samples = 300
    
    fig = plt.figure(figsize = (10, 10))
    
    plt.subplot(2,1,1)
    f  = h_sigmoid1
    fU = va.sampler.Apply(U, f, vectorized = True)
    data = fU(nb_samples)
    va.plot.spline_1d(fig.gca(), f, 0, 1, 200)
    plt.scatter(data['from_samples'], np.zeros(nb_samples), alpha = .1)
    plt.scatter(np.zeros(nb_samples), data['samples'],      alpha = .1)
    
    plt.subplot(2,1,2)
    f  = h_sigmoid2
    fU = va.sampler.Apply(U, f, vectorized = True)
    data = fU(nb_samples)
    va.plot.function_1d(fig.gca(), f, 0, 1, 200)
    plt.scatter(data['from_samples'], np.zeros(nb_samples), alpha = .1)
    plt.scatter(np.zeros(nb_samples), data['samples'],      alpha = .1)
    plt.show()

def test_005():
    U  = va.sampler.Uniform()
    f  = h_sigmoid1
    fU = va.sampler.Apply(U, f, vectorized = True)
    nb_samples = 1000
    offset = .05
    
    data = fU(nb_samples)
    samples = data['samples']
    from_samples = data['from_samples']
    pU  = va.estimator.density_1d(from_samples, inf=0, sup=1, bin_width=.02)
    pfU = va.estimator.density_1d(samples,      inf=0, sup=1, bin_width=.02)

    
    fig = plt.figure(figsize = (10, 10))
    gs  = fig.add_gridspec(2, 2,
                           width_ratios  = [2, 1],
                           height_ratios = [2, 1],
                           wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[0, 0])
    ax_main = ax
    ax.set_aspect('equal')
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    va.plot.spline_1d(ax, f, 0, 1, 200)
    ax.scatter(from_samples, np.zeros(nb_samples) + offset, alpha = .01)
    ax.scatter(np.ones(nb_samples) - offset, samples,       alpha = .01)
    
    ax = fig.add_subplot(gs[1, 0], sharex = ax_main)
    ax.set_xlim((0.0, 1.0))
    va.plot.function_1d(ax, pU, 0, 1, 200)
    
    ax = fig.add_subplot(gs[0, 1], sharey = ax_main)
    ax.set_ylim((0.0, 1.0))
    va.plot.function_1d(ax, pfU, 0, 1, 200, orientation = 'yx')
    
    plt.show()
    
        
def test_006():
    
    U  = va.sampler.Uniform()
    nb_samples = 300
    
    fig = plt.figure(figsize = (10, 5))
    
    f  = lambda x : .5*np.sin(2*np.pi*x) + x
    fU = va.sampler.Apply(U, f, vectorized = True)
    inf, sup = .5, .7
    va.plot.proba_1d(fig.gca(), fU, 0, 1, 1000,
                     lambda x, inf=inf, sup=sup : inf <= x <= sup,
                     'X', f'X \\in [{inf}, {sup}]')
    plt.show()


if __name__ == "__main__":
    
    if len(sys.argv) != 2 :
        print('usage : {} <test-number>'.format(sys.argv[0]))
        print()
        sys.exit(1)

    id = int(sys.argv[1])
    locals()['test_{:03d}'.format(id)]()


    
