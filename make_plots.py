"""
Script to create plots from simulation results
"""

from src.plot_functions import plot_grid

import matplotlib.pyplot as plt
import pandas as pd
import yaml


output_path = "results/figures/"
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    output_path = cfg["figures_dir"]
    results_dir = cfg.get("results_dir", "results/")
    results = pd.read_csv(results_dir + "raw/simulation_results.csv")
    
    plt.rcParams.update(cfg['rcparams'])
    
    colors = cfg['line_colors']
    linestyles = cfg['linestyles']

    print("Generating plot...")
    plot_grid(
        results=results,
        x_axis="m",
        y_axis="Power",
        factors=["method", "scheme", "m0_fraction"],
        se_bands=True,
        height=2,
        log_y_axis=False,
        log_x_axis=False,
        save_path=output_path + "power_grid",
        colors=colors,
        linestyles=linestyles,
    )
