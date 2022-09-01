import numpy as np

class Sampler:
    """
    Base class for sampler
    """
    def __init__(self):
        pass

    def __call__(self, nb_samples):
        """
        nb_samples : the number of samples you want
        returns {'samples' : [s1, s2, ... sn]}
        """
        return {'samples': np.array([])}

class Uniform(Sampler):
    """
    Uniform 1D float random variable in [inf, sup[.
    """
    def __init__(self, inf=0, sup=1):
        self.inf = inf
        self.sup = sup

    def __call__(self, nb_samples):
        return {'samples': np.random.rand(nb_samples) * (self.sup - self.inf) + self.inf,
                'bounds' : (self.inf, self.sup)}

class Nuplet(Sampler):
    """
    Builds a n-sized i.i.d sampler
    """
    def __init__(self, from_sampler, N):
        self.N = N
        self.from_sampler = from_sampler
        
    def __call__(self, nb_samples):
        return {'samples': np.array([self.from_sampler(self.N)['samples'] for i in range(nb_samples)]),
                'from_sampler' : self.from_sampler,
                'N' : self.N}


    


    
