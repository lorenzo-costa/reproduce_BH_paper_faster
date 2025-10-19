"""
Script to run the simulation study
"""
from src.simulation_functs import run_simulation
from src.metrics import Power, TrueRejections, RejectionsNumber
from src.methods import Bonferroni, BonferroniHochberg, FalseDiscoveryRate
import pickle

import numpy as np


if __name__ == "__main__":
    nsim = 100
    methods = [Bonferroni(), BonferroniHochberg(), FalseDiscoveryRate()]
    alpha = 0.05
    m = [4, 8, 16, 32, 64]
    m0 = [3/4, 1/2, 1/4, 0]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = [5, 10]
    scheme=["E", "I", "D"]
    
    rng = np.random.default_rng(10)
    sim_out, samples_list = run_simulation(nsim=nsim, 
                             m=m, 
                             m0_fraction=m0, 
                             L=L, 
                             scheme=scheme, 
                             method=methods, 
                             alpha=alpha, 
                             rng=rng, 
                             metrics=metrics)
    
    sim_out.to_csv("results/raw/simulation_results.csv", index=False)
    with open("data/simulated/simulation_samples.pkl", "wb") as f:
        pickle.dump(samples_list, f)