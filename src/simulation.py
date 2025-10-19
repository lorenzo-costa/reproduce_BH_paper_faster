from dgps import NormalGenerator, generate_means, compute_p_values
import numpy as np
import pandas as pd
import itertools


def run_scenario(samples, m0_fraction, L, scheme, method, alpha, metrics, rng=None):
    m = samples.shape[0]
    m0 = int(m*m0_fraction)
    means = generate_means(m=m, m0=m0, scheme=scheme, L=L, rng=rng)
    # uses property of Gaussian X ~ N(mu, 1) => X = mu + Z, Z ~ N(0,1)
    shifted_samples = samples + means
    p_values = compute_p_values(shifted_samples)
    rejected = method(p_values, alpha)
    
    results = {'m': m,
               'm0_fraction': m0_fraction,
               'm0': m0,
               'L': L,
               'scheme': scheme,
               'method': method.name}
    
    for eval_metric in metrics:
        results[eval_metric.name] = eval_metric(rejected, means)

    return results


def run_simulation(m, m0_fraction, L, scheme, method, alpha, metrics=None, nsim=100, rng=None):
    """Run simulation study for all combinations of parameters.

    Parameters
    ----------
    m : list or np.ndarray of int
        Number of hypotheses tested
    m0 : list or np.ndarray of float
        Fraction of true null hypotheses
    L : list or np.ndarray of int
        Upper bound on non-zero means
    scheme : list or np.ndarray of str
        Testing scheme to use
    method : list or np.ndarray of MultipleTesting
        Multiple testing correction methods to apply
    alpha : float
        Significance level
    nsim : int, optional
        Number of simulations to run, by default 100
    rng : np.random.Generator, optional
        Random number generator, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing simulation results for all scenarios
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if metrics is None:
        raise ValueError("At least one metric must be provided.")
    
    if not isinstance(m, (list, np.ndarray)):
        m = [m]
    if not isinstance(m0_fraction, (list, np.ndarray)):
        m0_fraction = [m0_fraction]
    if not isinstance(L, (list, np.ndarray)):
        L = [L]
    if not isinstance(scheme, (list, np.ndarray)):
        scheme = [scheme]
    if not isinstance(method, (list, np.ndarray)):
        method = [method]

    out = pd.DataFrame()
    for i in range(nsim):
        for m_i in m:
            samples = NormalGenerator(loc=0, scale=1).generate(m_i, rng=rng)
            for m0_i, L_i, scheme_i, method_i in itertools.product(m0_fraction, L, scheme, method):
                scenario_out = run_scenario(samples=samples, 
                                            m0_fraction=m0_i, 
                                            L=L_i, 
                                            scheme=scheme_i, 
                                            method=method_i, 
                                            alpha=alpha, 
                                            metrics=metrics, 
                                            rng=rng)
                scenario_out['nsim'] = i + 1
                out = pd.concat([out, pd.DataFrame(scenario_out, index=[0])], ignore_index=True)
    
    return out
    