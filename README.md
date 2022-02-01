# *Quantifying the uncertainty of decisions made through crop model simulations* experiment code

This respository contains the code used for the experiments of *Quantifying the uncertainty of decisions made from crop-model* based on the experiment from [[1]](#1), using the [DSSAT model](https://dssat.net/).

## Files used in the experiments
#### ```./dssat_files/```
Contains the DSSAT files used to replicate the long term maize experiment from [[1]](#1). Only required if DSSAT sampling to be replicated.

#### ```./dssat_samples/```
The simulations (as Python pickle files) used in this experiment using DSSAT's WGEN internal stochastic weather simulator.

#### ```./evaluation/```
The evaluation csv file used to evaluate DSSAT model performances at calibration stage. The data is for the conventional tillage, no residue incorporation treatment from [[1]](#1).

#### ```./output/```
The pre-computed (as pickle files) bound values for plots.

## Custom library prerequirements
#### 1. ```concentration_lib``` library (compulsory)
Libraries to compute concentration inequalities.

You can install (supposing Python >= 3.6) the ```concentration_lib``` running:
```bash
pip3 install concentration_lib
```
#### 2. ```gym-dssat``` bandit environment and ```dssatUtils``` libraries (optional)
Libraries to compute samples from DSSAT.

You can found source code and installation instructions at [https://github.com/rgautron/DssatBanditEnv](https://github.com/rgautron/DssatBanditEnv).
## Scripts of the experiments
To run a script, you can use:
 
```bash
python3 name_of_the_script.py
```

#### ```./dssat_sampling.py```
The Python file to get DSSAT samples using DSSAT's WGEN internal stochastic weather simulator. To launch this script, you will need to install the ```dssatUtils``` and ```gym_dssat``` custom libraries [introduced before]().

Note:
- in any case, we provide in ```./dssat_samples/``` the pre-computed samples (as pickle files) used in the experiments: it does not require running ```dssat_sampling.py``` before running the other scripts.
- if you need to run DSSAT's sampling for this experiment, you must copy ```./dssat_files/MZCER047.CUL``` to ```/Genotype/``` in DSSAT's root folder.
#### ```./plot_bounds.py```
The Python file to compute, store/load and plot the mean and variance confidence intervals presented in the paper.

Note that bound computation is computationally intensive: we provide pre-computed values in case you do not want to compute them.

#### ```./plot_dists.py```
The Python file to plot the distributions sampled from DSSAT.

#### ```./plot_error_bars.py```
The Python file to find the interval disjunction minimal risk level and to plot the resulting confidence intervals as error bars.

#### ```./plot_evaluation.py```
The Python file to plot residue inspection for model evaluation stage, and error/simulated distributions qq-plots.

#### ```./statistical_tests.py```
The Python file for the statistical tests for the model error distribution.

#### ```./utils.py```
Utilitary functions used in other Python files.

#### ```./dssat_hypothesis_testing.ipynb```
Python Jupyter Notebook of the simulated yield distribution hypothesis testing for normality. Corresponds to the paper's appendix Section D.

## References
<a id="joshi">[1]</a> Joshi, N., Singh, A. K., & Madramootoo, C. A. (2017). 
Application of DSSAT model to simulate corn yield under long-term tillage and residue practices. Transactions of the ASABE, 60(1), 67-83.