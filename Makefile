# Makefile

# Variables
PYTHON=python
SRC=src
DATA=data
RESULTS=results

# Default target
.PHONY: all simulate figures clean

all: simulate figures

# Step 1: Run simulations
simulate:
	$(PYTHON) -m run_simulation

# Step 2: Generate figures (optional if scripts already output plots)
figures:
	$(PYTHON) -m make_plots
	@echo "Figures should now be in $(RESULTS)/figures"

# Clean up caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest