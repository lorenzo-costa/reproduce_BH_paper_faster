"""Data generation classes for simulation studies.

This module provides a generator for data from a Gaussian distribution as required 
by the experiments. This is implemented to make the
simulations scalable and extensible to other distributions.
"""

import numpy as np
from abc import ABC, abstractmethod

from scipy import stats


class DataGenerator(ABC):
    """Abstract base class for data generation.
    
    All data generators must implement three methods:
    - generate(n): Create n samples from the distribution
    - name(): Return a descriptive name for reporting
    
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
    
    def generate(self, n=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if n is None:
            n = self.loc.shape[0] if isinstance(self.loc, np.ndarray) else 1
        return rng.normal(self.loc, self.scale, n)
    
    @property
    def name(self):
        return f"Normal(mu={self.loc}, sigma={self.scale})"
    
    @property
    def null_value(self):
        return self.loc


def generate_means(m, m0, scheme, L, rng=None):
    """Generate a simulation scenario from a Gaussian sample.

    Parameters
    ----------
    m : int
        Number of hypotheses to test
    m0 : int
        Number of true null hypotheses
    scheme : str
        The testing scheme to use ('E', 'D', or 'I')
    L : int
        Non-zero mean upper bound
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng()
    Returns
    -------
    np.ndarray
        Array of means for the m hypotheses
        
    Examples
    --------
    >>> means = generate_means(m=4, m0=2, scheme='E', L=4, rng=np.random.default_rng(42))
    >>> means
    array([0., 0., 1., 1.])
    >>> means = generate_means(m=4, m0=2, scheme='D', L=8, rng=np.random.default_rng(42))
    >>> means
    array([0., 0., 6., 8.])
    
    """
    
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer")
    if not isinstance(m0, int) or m0 < 0 or m0 > m:
        raise ValueError("m0 must be an integer between 0 and m inclusive")
    if rng is None:
        rng = np.random.default_rng()

    if scheme == 'E':
        weights = np.array([1, 1, 1, 1])
    elif scheme == 'D':
        weights = np.array([4, 3, 2, 1])
    elif scheme == 'I':
        weights = np.array([1, 2, 3, 4])
    else:
        raise ValueError("Invalid scheme. Choose from ['E', 'D', 'I']")
    
    m1 = m - m0
    counts = (weights / weights.sum()) * m1
    counts = np.round(counts).astype(int)
    # adjust for rounding errors
    diff = m1 - counts.sum()
    counts[0] += diff
    
    levels = [L/4, L/2, 3*L/4, L]
    means = np.concatenate([
        np.zeros(m - m1),
        np.repeat(levels, counts)])
    np.random.shuffle(means)
    
    return means

def compute_p_values(normal_samples):
    return 2 * stats.norm.cdf(-np.abs(normal_samples))

def generate_scenario(m, m0, scheme, L, rng=None):
    """Generate a simulation scenario with p-values from Gaussian samples.

    Parameters
    ----------
    m : int
        Number of hypotheses to test
    m0 : int
        Number of true null hypotheses
    scheme : str
        The testing scheme to use ('E', 'D', or 'I')
    L : int
        Non-zero mean upper bound
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng()

    Returns
    -------
    np.ndarray
        Array of p-values for the m hypotheses
    """
    if rng is None:
        rng = np.random.default_rng()
        
    means = generate_means(sample=None, m=m, m0=m0, scheme=scheme, L=L, rng=rng)
    samples = rng.normal(loc=means, scale=1.0, size=m)
    p_values = compute_p_values(samples)
    return p_values
