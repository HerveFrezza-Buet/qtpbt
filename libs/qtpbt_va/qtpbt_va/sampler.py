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
        super().__init__()
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
        super().__init__()
        self.N = N
        self.from_sampler = from_sampler
        
    def __call__(self, nb_samples):
        return {'samples': np.array([self.from_sampler(self.N)['samples'] for i in range(nb_samples)]),
                'from_sampler' : self.from_sampler,
                'N' : self.N}

class Apply(Sampler):
    """
    Builds f(U) from U.
    """
    def __init__(self, from_sampler, f, vectorized = False):
        """
        f is applied to each sample. If f can be applied directly to the array of samples, provide vectorized=True.
        """
        super().__init__()
        self.f = f
        self.from_sampler = from_sampler
        self.vectorized = vectorized
        
    def __call__(self, nb_samples):
        from_samples = self.from_sampler(nb_samples)['samples']
        if self.vectorized :
            samples = self.f(from_samples)
        else:
            samples = np.array([self.f(x) for x in from_samples])
        return {'samples': samples,
                'from_sampler' : self.from_sampler,
                'f' : self.f,
                'from_samples' : from_samples}

        
        

    


    
