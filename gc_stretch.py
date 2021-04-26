import argparse
import time
from itertools import zip_longest
import numpy as np
import cooler

import hic_analysis as hic
import gc_datafile
import gc_model

def stretch_chr(lambdas, ratio, target_nn=None):
    target_length = None if target_nn is None else target_nn.shape[0]
    target_slice = slice(None, target_length)
    stretched = np.repeat(lambdas, ratio, axis=0)[target_slice]
    if target_nn is not None:
        stretched[~target_nn] = np.nan
        # Fill in bins that used to be NaN and now are not with the first previous non-nan bin
        prev = [1] + [0] * (lambdas.shape[1]-1) # A default value for the first bin if it's nan: [1, 0, 0, ...]
        for idx in np.nonzero(target_nn)[0]:
            if np.isnan(stretched[idx]).any():
                stretched[idx] = prev
            else:
                prev = stretched[idx]
    return stretched


def stretch_fit(lowres_params, ratio, target_nn_list=None):
    """
    Stretch state_probabilities of a fit by ratio. If target_nn_list is given, it should be
    a list of masks, one per chromosome, giving the _high resolution_ bins that are not NaN.

    The return value is the model parameters with the updated state_probabilities and cis_lengths.
    state_probabilities are truncated to the length of the non-NaN mask.
    """
    per_chr_lambdas = np.split(lowres_params['state_probabilities'],
                               np.cumsum(lowres_params['cis_lengths']))[:-1]
    stretched_lambdas = [ stretch_chr(l, ratio, nn) for l, nn in zip_longest(per_chr_lambdas, target_nn_list or []) ]
    cis_lengths = [ l.shape[0] for l in stretched_lambdas ]

    model_params = { **lowres_params,
                    'state_probabilities': np.concatenate(stretched_lambdas),
                    'cis_lengths': cis_lengths }

    return model_params


def get_nn(mcool_filename, resolution, chromosome):
    c = cooler.Cooler(f'{mcool_filename}::/resolutions/{resolution}')
    if chromosome == 'all':
        weights = c.bins()[:]['weight']
    else:
        chr_start, chr_end = c.extent(chromosome)
        weights = c.bins()[chr_start:chr_end]['weight']
    return ~np.isnan(weights.to_numpy())


def main():
    parser = argparse.ArgumentParser(description = 'Stretch low-res solutions to higher-res')
    parser.add_argument('-f', help='fit to stretch', dest='filename', type=str, required=True)
    parser.add_argument('-o', help="Output file", dest='output_filename', required=True)
    parser.add_argument('--source-res', help='Source resolution', dest='source_res', type=int, required=True)
    parser.add_argument('--target-res', help='Target resolution', dest='target_res', type=int, required=True)
    parser.add_argument('--chroms', help='Manually specify chromosomes', dest='chrom', type=str, required=False)
    parser.add_argument('--mcool', help='Manually specify mcool', dest='mcool', type=str, required=False)
    args = parser.parse_args()

    if args.source_res % args.target_res != 0:
        print("Stretching is only supported to a resolution that is an integer-multiple of the source resolution")
        sys.exit(1)
    print("Reading fit from file:", args.filename)
    fit = gc_datafile.load(args.filename)
    chrom = args.chrom or fit['metadata']['args']['chrom']
    mcool_filename = args.mcool or fit['metadata']['args']['filename']
    print(f"Original mcool filename is: {mcool_filename}")
    if chrom == 'all':
        chromnames = cooler.Cooler(f"{mcool_filename}::/resolutions/{args.target_res}").chromnames
    else:
        chromnames = [ x if x.startswith('chr') else f'chr{x}' for x in chrom.split(',') ]
    print(f"Working on chromosomes: {chromnames}")

    ratio = args.source_res // args.target_res
    print(f"Stretching solution from {args.source_res} to {args.target_res}, ratio is {ratio}")
    target_nn_list = [ get_nn(mcool_filename, args.target_res, c) for c in chromnames ]

    parameters = stretch_fit(fit['parameters'], ratio, target_nn_list)
    metadata = dict(lowres_metadata = fit['metadata'], stretch_args = vars(args))
    
    print(f'Saving into {args.output_filename}')
    gc_datafile.save(args.output_filename, parameters=parameters, metadata=metadata)
    print(f'Done.')

if __name__ == '__main__':
    main()

