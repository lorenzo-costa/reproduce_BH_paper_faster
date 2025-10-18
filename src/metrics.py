"""Evaluation metrics for simulation studies."""

import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract base class for evaluation metrics.
    
    All metrics must implement two methods:
    - __call__(rejected, true_values): Compute the metric given rejections and true values
    - name(): Return a descriptive name for reporting
    """
    
    @abstractmethod
    def __call__(self, rejected, true_values):
        """Compute the evaluation metric.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        float
            Computed metric value
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the metric.

        Returns
        -------
        str
            Human-readable name of the metric
        """
        pass

class Power(Metric):
    """Statistical power metric.

    Power is defined as the proportion of true alternative hypotheses
    that are correctly rejected.
    """

    @property
    def name(self):
        return "Statistical Power"

    def __call__(self, rejected, true_values):
        """Compute the statistical power.

        Power is defined as the proportion of true alternative hypotheses
        that are correctly rejected.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        float
            Statistical power
        """
        truth_mask = (true_values != 0)
        power = np.mean(rejected[truth_mask]) if np.sum(truth_mask) > 0 else 0.0
        
        return power