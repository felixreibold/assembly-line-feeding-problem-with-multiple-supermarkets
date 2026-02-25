# Assembly Line Feeding Problem with Multiple Supermarkets (MILP)

Extension of the Assembly Line Feeding Problem model by Adenipekun et al. (2022), integrating multiple supermarkets into the existing MILP framework. The model optimizes feeding policies, vehicle types, routes, and supermarket allocation to minimize total logistics costs. Implemented in Python using Gurobi and evaluated via numerical experiments.

This repository contains the full implementation of the MILP model developed in our seminar paper:

[Final Report (PDF)](final-report-assembly-line-feeding-problem.pdf)

The model extends the Assembly Line Feeding Problem (ALFP) by introducing multiple supermarkets and integrating tactical supermarket assignment decisions into the existing cost-minimization framework.

The implementation is written in Python using Gurobi.

---

## Model Overview

The model simultaneously determines:

- Feeding policies per part
- Vehicle type selection
- Route assignment
- Fleet sizing
- Assignment of parts to supermarkets
- Activation and capacity of multiple supermarkets

Objective: minimize total feeding costs including transportation, replenishment, preparation, usage, and supermarket operating costs.

For full theoretical background, formulation, and numerical experiments, please refer to the included paper.

---

## Requirements
- Python 3.10+
- Gurobi (licensed)
- Install dependencies:
  pip install -r requirements.txt

## Data
Datasets are algorithmically generated and correspond to the experimental design described in the paper.

- data/Input_data/: Input_Data.xlsx, n_pm.xlsx, l_pm.xlsx, n_f.xlsx
- data/datasets/: distance matrices, part-station assignments, experimental plan

## Run (example)
(Entry-point script will be added in src/.)

## Authors

Felix Reibold (M.Sc.)
Steffen Voigtländer (M.Sc.)
