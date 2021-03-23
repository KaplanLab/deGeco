import argparse
import numpy as np

def convert(old_fit, cis_lengths=None):
    parameters = dict(state_probabilities=old_fit['lambdas'],
                      state_weights=old_fit['weights'],
                      cis_dd_power=old_fit['alpha'],
                      trans_dd=old_fit['beta'])
    if cis_lengths is None:
        parameters['cis_lengths'] = [old_fit['lambdas'].shape[0]]
    else:
        parameters['cis_lengths'] = cis_lengths
    metadata = dict(seed=old_fit['seed'])

    return dict(parameters=parameters, metadata=metadata)

def main():
    parser = argparse.ArgumentParser(description = 'Convert old format fit to new format fit')
    parser.add_argument('-f', help='old fit filename', dest='filename', type=str, required=True)
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=True)
    parser.add_argument('--lengths', help='override cis_lengths', dest='cis_lengths', type=int, nargs='+', required=False, default=None)
    args = parser.parse_args()

    print("Reading old fit from file:", args.filename)
    old_fit = np.load(args.filename)
    print("Converting")
    new_fit = convert(old_fit, args.cis_lengths)
    print(f'Saving into {args.output}')
    np.savez_compressed(args.output, **new_fit)
    print(f'Done.')

if __name__ == '__main__':
    main()
