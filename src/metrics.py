import numpy as np

def compute_power(rejected, true_values):
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