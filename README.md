# Compartments

A probability-based model of genome compartments, used to explain results of Hi-C experiments.

The model supports any number of compartments and allows for intra- and inter-compartment interaction
with different weights.

## Summary of files

Exeuctables:

* `gc_model_main.py` is the executable used to fit the model to a given Hi-C data file. Data can be in mcool or npy formats.
* `visualize.py` is an executable used to visualize the results of `gc_model_main`
* `figures.py` is another visualization script, showing different figures
* `*_script.sh` are wrapper scripts to be used when running on clusters

Libraries:

* `gc_model.py` implements the model. It has two main methods:
    * `fit` method, used by `gc_model_main` to fit model to data
    * `generate_interaction_matrix` used to reconstruct a Hi-C data matrix using the model parameters
* `hic_analysis.py` is a library of useful functions to analyse and process Hi-C data. Used by all other files.
