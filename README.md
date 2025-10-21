# Reproducing Benjamini & Hochberg (1995)

## Project Description

This project aims to reproduce the findings from the paper "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing" by Yoav Benjamini and Yosef Hochberg (1995). The analysis involves simulating multiple hypothesis tests to compare the statistical power and error control of the Benjamini-Hochberg (BH) procedure against traditional Bonferroni-type methods.

## Setup Instructions

You can set up the necessary environment using either Conda (recommended, what I'm using) or pip.

### Option 1: Using Conda

1.  Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
2.  Create the Conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
3.  Activate the newly created environment:
    ```bash
    conda activate bh-repro
    ```

### Option 2: Using pip

1.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  Install the required packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Analysis

This project uses a `Makefile` to streamline the simulation and analysis pipeline.

* **To run the entire analysis from start to finish:**
    ```bash
    make all
    ```
    This command will execute the simulation, perform the analysis, and generate the final figures.

* **To run individual steps:**
    * `make simulate`: Runs only the data simulation.
    * `make analyse`: Runs the analysis on the simulated data.
    * `make figures`: Generates the figures from the analyzed data.

## Estimated Runtime

The estimated time to run the complete analysis (`make all`) is approximately **2351 seconds** with a MacBook Air with Apple M4 chip and 16GB of RAM.

## Summary of Key Findings

The results of this reproduction confirm that the Benjamini-Hochberg (BH) procedure offers substantially more statistical power than traditional Bonferroni-type procedures. This advantage becomes more pronounced as the number of tested hypotheses (`m`) increases, while the control of the False Discovery Rate (FDR) is less conservative than the overly strict Family-Wise Error Rate (FWER) control imposed by Bonferroni-type methods.

