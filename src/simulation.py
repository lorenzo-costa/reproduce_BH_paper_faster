from dgps import NormalGenerator, generate_means, compute_p_values
import numpy as np
import pandas as pd

def run_simulation(nsim, m_list, m0_list, L_list, method_list, alpha, rng=None):
    out = pd.DataFrame()
    for i in range(nsim):
        for m in m_list:
            samples = NormalGenerator(loc=0, scale=1).generate(m, rng=rng)
            for m0 in m0_list:
                for L in L_list:
                    for scheme in ["E", "D", "I"]:
                        means = generate_means(m=m, m0=m0, scheme=scheme, L=L, rng=rng)
                        true_mask = (means != 0)
                        shifted_samples = samples + means
                        p_values = compute_p_values(shifted_samples)
                        for method in method_list:
                            rejected = method(p_values, alpha)
                            true_rejections = np.sum(rejected[true_mask])
                            out = pd.concat([out, pd.DataFrame({
                                'nsim': i+1,
                                'm': m,
                                'm0': m0,
                                'L': L,
                                'scheme': scheme,
                                'method': method.name,
                                'true_rejections': true_rejections,
                                'n_rejected': rejected.sum()
                            }, index=[0])], ignore_index=True)            
    return out
        
    