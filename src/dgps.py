"""Data generation classes for simulation studies.

This module provides a generator for data from a Gaussian distribution as required 
by the experiments. This is implemented to make the
simulations scalable and extensible to other distributions.
"""

import numpy as np
from abc import ABC, abstractmethod


class DataGenerator(ABC):
    """Abstract base class for data generation.
    
    All data generators must implement three methods:
    - generate(n): Create n samples from the distribution
    - name(): Return a descriptive name for reporting
    - null_value(): Return the true parameter value under H0
    
    This ensures all generators can be used interchangeably in simulation
    studies without modifying the simulation code.
    """
    
    @abstractmethod
    def generate(self, n, rng=None):
        """Generate n samples from the distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng (no seed) 
        
        Returns
        -------
        np.ndarray
            Array of n samples from the distribution
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the distribution.
        
        Returns
        -------
        str
            Human-readable name including parameter values
        """
        pass
    
    @property
    @abstractmethod
    def null_value(self):
        """Return the true parameter value under H0.
        
        For testing whether a location parameter equals a specific value,
        this returns that true value. For example, if testing H0: μ = μ0,
        this returns μ0.
        
        Returns
        -------
        float
            True value of the location parameter
        """
        pass


class NormalGenerator(DataGenerator):
    """Generate data from a normal distribution.
    
    The normal distribution is symmetric and serves as the baseline case
    for most parametric tests. It is characterized by its mean (loc) and
    standard deviation (scale).
    
    Parameters
    ----------
    loc : float, default=0
        Mean of the distribution (location parameter)
    scale : float, default=1
        Standard deviation of the distribution (scale parameter)
    
    Examples
    --------
    >>> gen = NormalGenerator(loc=5, scale=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'Normal(mu=5, sigma=2)'
    """
    
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.loc, self.scale, n)
    
    @property
    def name(self):
        return f"Normal(mu={self.loc}, sigma={self.scale})"
    
    @property
    def null_value(self):
        return self.loc