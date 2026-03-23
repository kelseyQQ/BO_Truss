# BO_Truss

Bayesian optimization for constrained truss design.

This repository contains code for optimizing truss structures using **Bayesian Optimization (BO)** together with **Gaussian Process (GP)** surrogate models. The framework is designed for constrained structural optimization problems, where the goal is to reduce structural weight while satisfying requirements such as stress, buckling, displacement, and frequency constraints.

## Overview

In structural design, evaluating each candidate design often requires a finite element analysis (FEA), which can be computationally expensive. This repository explores how Bayesian optimization can improve search efficiency by learning from previous simulations and selecting promising new designs.

The code focuses on truss optimization problems in which both:
- **member sizes** and/or
- **nodal geometric variables**

can be treated as design variables.

The optimization aims to:
- minimize structural weight,
- satisfy structural constraints,
- reduce the number of expensive analyses.

## Features

- Bayesian optimization for constrained design problems
- Gaussian Process surrogate models for objective and constraints
- Support for multiple GP settings
- Investigation of different initialization strategies
- Geometry-informed bias in initial sampling
- Multi-seed runs for statistical comparison
- Truss problem definitions using `.pro` and `.geom` files

## Repository Structure

A typical structure of the repository includes:

- `run_18GP_multi_seed.py`  
  Script for running multiple optimization trials with different random seeds.

- `measure_weight.py`  
  Utility for computing structural weight.

- `cantilever.pro`, `cantilever.geom`  
  Problem definition files for the truss model.

- `cantilever_36GP.pro`, `cantilever_36GP.geom`  
  Alternative truss problem settings.

- `cantilever_measure_weight.pro`, `cantilever_measure_weight.geom`  
  Files related to weight evaluation.

- `pyJive/`  
  External or local framework dependency used for structural analysis.

## Methodology

The workflow generally consists of the following steps:

1. Generate an initial set of design samples  
2. Evaluate these samples using structural analysis  
3. Train GP surrogate models for the objective and constraints  
4. Use a constrained acquisition function to select the next candidate  
5. Repeat until the computational budget is exhausted

In this repository, special attention is given to the effect of:

- **geometry-informed bias**, and
- **large initial sampling followed by selection of better initial points**

on the convergence behavior of BO.

## Requirements

This project is written in Python and typically uses:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pyJive` or related structural analysis tools

Depending on your environment, additional packages may be required.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/kelseyQQ/BO_Truss.git
cd BO_Truss
