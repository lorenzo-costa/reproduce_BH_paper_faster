"""
Script to run the simulation study
"""

from src.simulation_functs import run_simulation
from src.metrics import Power, TrueRejections, RejectionsNumber
from src.methods import Bonferroni, BonferroniHochberg, BenjaminiHochberg

import pickle
import numpy as np
import yaml
import time

method_map = {
    "Bonferroni": Bonferroni,
    "BonferroniHochberg": BonferroniHochberg,
    "BenjaminiHochberg": BenjaminiHochberg,
}

if __name__ == "__main__":
    # load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    nsim = cfg["nsim"]
    methods = [method_map[name]() for name in cfg["methods"]]
    alpha = cfg["alpha"]
    m = cfg["m"]
    m0 = cfg["m0"]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = cfg["L"]
    scheme = cfg["scheme"]
    rng = np.random.default_rng(cfg["rng_seed"])

    results_dir = cfg.get("results_dir", "results/")
    data_dir = cfg.get("data_dir", "data/")

    start_time = time.time()
    sim_out, samples_list = run_simulation(
        nsim=nsim,
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        rng=rng,
        metrics=metrics,
        results_dir=results_dir,
        parallel=True,
    )

    sim_out.to_csv(f"{results_dir}/raw/full_simulation_results.csv", index=False)
    with open(f"{data_dir}/simulated/simulation_samples.pkl", "wb") as f:
        pickle.dump(samples_list, f)
    end = time.time()
    print(f"Simulation completed in {end - start_time:.2f} seconds.")
