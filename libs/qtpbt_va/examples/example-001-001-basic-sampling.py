
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
h_sigmoid = va.utils.Spline(Xs, Ys, degree=3) 


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
    f  = h_sigmoid
    fU = va.sampler.Apply(U, f, vectorized = True)
    fig = plt.figure(figsize = (10, 5))
    nb_samples = 300
    data = fU(nb_samples)
    va.plot.spline_1d(fig.gca(), f, 0, 1, 200)
    plt.scatter(data['from_samples'], np.zeros(nb_samples), alpha = .1)
    plt.scatter(np.zeros(nb_samples), data['samples'],      alpha = .1)
    plt.show()

def test_005():
    U  = va.sampler.Uniform()
    f  = h_sigmoid
    fU = va.sampler.Apply(U, f, vectorized = True)
    fig = plt.figure(figsize = (10, 5))
    nb_samples = 1000
    samples = fU(nb_samples)['samples']
    p = va.estimator.density_1d(samples)
    plt.scatter(samples, np.zeros(nb_samples), alpha = .1)
    va.plot.function_1d(fig.gca(), p, 0, 1, 200, vectorized = True)
    plt.show()
    
                          


if __name__ == "__main__":
    
    if len(sys.argv) != 2 :
        print('usage : {} <test-number>'.format(sys.argv[0]))
        print()
        sys.exit(1)

    id = int(sys.argv[1])
    locals()['test_{:03d}'.format(id)]()


    
