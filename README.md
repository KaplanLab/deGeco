# deGeco

A probability-based model of genome compartments, used to explain results of Hi-C experiments.

The model supports any number of compartments and allows for intra- and inter-compartment interaction
with different weights.

Dataset of whole-genome fits of various cell types can be found in [Zenodo](https://zenodo.org/record/7152655).

## Summary of files

Exeuctables:

* `gc_model_main.py` is the executable used to fit the model to a given Hi-C data file. Data can be in mcool or npy formats.
* `gc_datafile.py` provides a nice interface for reading and writing output files. Can also be used as a module.
* `get_best.py` finds the best fit from the outputs of several iterations.
* `downsample.py` Samples reads (without replacement) from a Hi-C map
* `resample.py` takes a Hi-C map or a deGeco fit, treats them as interaction probability matrices and samples them with replacement.
* `scripts` dir contains scripts for analysing data, downsampling matrices, etc

Libraries:

* `gc_model.py` implements the model of compartments interactions. It has two main methods:
    * `fit` method, used by `gc_model_main` to fit model to data
    * `fit_sparse` method, uses the Cython implementation found in `loglikelihood.pyx` (for the model) and `zero_sampler.pyx` (for estimating the partition function)
    * `generate_interaction_matrix` used to reconstruct a Hi-C data matrix using the model parameters
* `distance_decay_model.py` implements the model of distance-related interactions: power-law (for cis) and background (for trans).
* `checkpoint.py` and `interruptible_lbfgs.py` implement checkpointing the LBFGS run.
* `hic_analysis.py`, `array_utils.py`, `model_utils.py` contain auxillary functions.

## Building

To build the Cython modules, install Cython (`pip install -r requirements.txt`) and run:

```
python setup.py build_ext -b .
```

## Fitting the model
The model accepts Hi-C data as `.mcool` or `.npy` files.  Fitted parameters will be written as an `.npz` file.

### Fitting small matrices (single chromosomes or all-genome in low resolution)
Small matrices can be simply fitted using the default 'dense' implementation, which loads the entire matrix to memory.  For example, to fit chromosome 19 at 20Kb from an mcool using 2 states and write to results to `chr19_2st_20000.npz` run:

```
python gc_model.py -m data.mcool -kb 20000 -ch chr19 -n 2 -o chr19_2st_20000.npz
```

This will run 10 iterations serially and choose the best one. To run in parallel, limit the amount of iterations run serially with the `--iterations` parameter and run several instances of `gc_model.py` in parallel. For reproducible results use the `--seed` parameter (which accepts multiple values, for each iteration run serially).

### Fitting large matrices with sparse multi-res fits
Large matrices should probably use the sparse implementation, and be initialized to previous low-resolution fit. See `scripts/stretch_pipeline.sh` for an impementation of this pipeline, or see below for an explanation of how it works.

Initializing to a lower resolution fit significantly reduces fit time, since the lower-resolution fit usually requires fewer optimization steps to converge than a random intialization. To use a low resolution fit, use the `--init` and `--init-stretch-by` parameters. For example, to initialize a 5Kb fit with a previous 20Kb run:

```
python gc_model.py -m data.mcool -kb 5000 -ch ch19 -n 4 -o chr19_4st_5000.npz --init chr19_4st_20000.npz --init-stretch-by 4
```

Using the sparse implementation dramatically reduces memory requirements. This model samples a subset of the zero entries in the Hi-C matrix and ignores the rest, saving CPU time and memory. To use it, use the `--sparse` parameter and specify how many zero pixels to sample using the `--zero-sample` parameter. Sampling about the same number of zero pixels as non-zero pixels seems to work well.

To fit the entire genome (without Y and M chromosomes) using the sparse implementation and initialization to a lower resolution fit, run:

```
python gc_model.py -m data.mcool -kb 50000 -ch all_noym -n 4 -o all_4st_50000.npz --init all_4st_250000.npz --init-stretch-by 5 --sparse --zero-sample 626412308 # zero-sample value taken from nnz entry in cooler info
```

The resulting fit can be used for another round of stretching, to get a higher resolution fit.

Note: the sparse implementation requires mcool files without duplicate entries in their `pixels` table. Using such mcool files may result in a warning with a negative number of zeros (`Trying to sample 12345 which is more than total zeors count -100, using total instead`) followed by an exception. Running `cooler zoomify` on the mcool with a new version of cooler fixes this issue.

## Output files
The output file format is NumPy's npz object that has two main keys:

1. `metadata` - an object containing various information on the run: command line parameters, duration of run, etc
2. `parameters` - an object containing the actual fitted parameters:
    1. `state_probabilities` - an NxS matrix of state probabilities, where N is the number of bins and S the number of states the model was run with
    2. `cis_weights` - an SxS matrix of cis state affinities
    3. `trans_weights` - an SxS matrix of trans state affinities
    4. `cis_dd_power` - the exponent of the power law decay of interaction intensity in cis (also denoted as alpha)
    5. `trans_dd` - the constant background level of trans interaction (also denoted as beta)
    6. `cis_lengths` - Number of bins for each chromosome. Sum of `cis_lengths` is N, the total number of bins.

To read using numpy:

```
import numpy as np

fit = np.load(filename, allow_pickle=True)
metadata = fit['metadata'][()]
parameters = fit['parameters'][()]
```

or use the `gc_datafile` module:

```
import gc_datafile
parameters = gc_datafile.load_params(filename)
```
