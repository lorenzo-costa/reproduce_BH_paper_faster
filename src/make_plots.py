"""
Script to create plots from simulation results
"""

from src.plot_functions import plot_individual, plot_grid
import pandas as pd

output_path = "results/figures/"
if __name__ == "__main__":
    results = pd.read_csv("results/raw/simulation_results.csv")
    
    plot_grid(
        results=results,
        x_axis="m",
        y_axis="Power",
        factors=["method", "scheme", "m0_fraction"],
        se_bands=True,
        height=2,
        log_y_axis=False,
        log_x_axis=False,
        save_path = output_path+"power_grid"
    )
