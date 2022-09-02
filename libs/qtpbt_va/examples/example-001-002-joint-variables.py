
import sys
import numpy as np
import qtpbt_va as va
import matplotlib.pyplot as plt

offset = .05
    
Xs  = np.array([0.00, 0.01, 0.05, 0.10, 0.20, 0.50])
Ys  = np.array([0.00, 0.10, 0.30, 0.35, 0.40, 0.50])
XXs = 1.0 - np.flip(Xs[:-1])
YYs = 1.0 - np.flip(Ys[:-1])
Xs  = np.hstack((Xs, XXs))
Ys  = np.hstack((Ys, YYs))
fI = va.utils.Spline(Xs, Ys, degree=3)

Xs  = Xs * .5
Ys  = Ys * .5
XXs = Xs[1:] + .5
YYs = Ys[1:] + .5
Xs  = np.hstack((Xs, XXs))
Ys  = np.hstack((Ys, YYs))
fJ  = va.utils.Spline(Xs, Ys, degree=3)

U = va.sampler.Uniform() 
I = va.sampler.Apply(U, fI, vectorized = True)
J = va.sampler.Apply(U, fJ, vectorized = True)

UUUU      = va.sampler.Nuplet(U, 4)
UUU       = va.sampler.Nuplet(U, 3)

def noisy(f, epsilon = .1):
    return lambda x, u : f(x) + epsilon * (2 * u - 1)
    
IJindep = va.sampler.Apply(UUUU, lambda u : (noisy(fI)(u[0], u[1]), noisy(fJ)(u[2], u[3])))

IJdep = va.sampler.Apply(UUU, lambda u : (noisy(fI)(u[0], u[1]), noisy(fJ)(u[0], u[2])))

def test_001():
    nb_samples = 300
    dataI = I(nb_samples)
    dataJ = J(nb_samples)

    fig = plt.figure(figsize = (10, 5))
    gs  = fig.add_gridspec(1, 2,
                           width_ratios  = [2, 2],
                           height_ratios = [1],
                           wspace=0.1, hspace=0.1)
    
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.set_title('Random variable I')
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    va.plot.function_1d(ax, fI, 0, 1, 200)
    ax.scatter(dataI['from_samples'], np.zeros(nb_samples) + offset, alpha = .05)
    ax.scatter(np.ones(nb_samples) - offset, dataI['samples'],       alpha = .05)
    
    ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect('equal')
    ax.set_title('Random variable J')
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    va.plot.function_1d(ax, fJ, 0, 1, 200)
    ax.scatter(dataJ['from_samples'], np.zeros(nb_samples) + offset, alpha = .05)
    ax.scatter(np.ones(nb_samples) - offset, dataJ['samples'],       alpha = .05)

    plt.show()

def test_002():
    nb_samples = 5000
    samples = IJindep(nb_samples)['samples']
    
    fig = plt.figure(figsize = (10,10))
    gs  = fig.add_gridspec(2, 2,
                           width_ratios  = [2, 1],
                           height_ratios = [2, 1],
                           wspace=0.2, hspace=0.2)

    va.plot.joint(fig.add_subplot(gs[0, 0]),
                  fig.add_subplot(gs[1, 0]),
                  fig.add_subplot(gs[0, 1]),
                  'I', 'J', samples, offset=offset)
    
    plt.show()

def test_003():
    nb_samples = 5000
    samples = IJdep(nb_samples)['samples']
    
    fig = plt.figure(figsize = (10,10))
    gs  = fig.add_gridspec(2, 2,
                           width_ratios  = [2, 1],
                           height_ratios = [2, 1],
                           wspace=0.2, hspace=0.2)

    va.plot.joint(fig.add_subplot(gs[0, 0]),
                  fig.add_subplot(gs[1, 0]),
                  fig.add_subplot(gs[0, 1]),
                  'I', 'J', samples, offset=offset)
    
    plt.show()


    
    


if __name__ == "__main__":
    
    if len(sys.argv) != 2 :
        print('usage : {} <test-number>'.format(sys.argv[0]))
        print()
        sys.exit(1)

    id = int(sys.argv[1])
    locals()['test_{:03d}'.format(id)]()
