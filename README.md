# deGeco

A probability-based model of genome compartments, used to explain results of Hi-C experiments.

The model supports any number of compartments and allows for intra- and inter-compartment interaction
with different weights.

## Summary of files

Exeuctables:

* `gc_model_main.py` is the executable used to fit the model to a given Hi-C data file. Data can be in mcool or npy formats.
* `scripts` dir contains scripts for analysing data, downsampling matrices, etc

Libraries:

* `gc_model.py` implements the model. It has two main methods:
    * `fit` method, used by `gc_model_main` to fit model to data
    * `generate_interaction_matrix` used to reconstruct a Hi-C data matrix using the model parameters
* `hic_analysis.py` is a library of useful functions to analyse and process Hi-C data. Used by all other files.
