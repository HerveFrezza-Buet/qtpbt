
import sys
import numpy as np
import qtpbt_va as va
import matplotlib.pyplot as plt


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



if __name__ == "__main__":
    
    if len(sys.argv) != 2 :
        print('usage : {} <test-number>'.format(sys.argv[0]))
        print()
        sys.exit(1)

    id = int(sys.argv[1])
    locals()['test_{:03d}'.format(id)]()


    
