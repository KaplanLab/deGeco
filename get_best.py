#!/usr/bin/python3
import argparse
import glob
import os
import numpy as np

import gc_model
import gc_datafile

def get_best(files, n=1, recursive=True, metadata_field='optimize_value'):
    sorted_fits = []
    for f in files:
        if os.path.isdir(f) and recursive:
            npz_files = glob.glob(f'{f}/*.npz')
            fits = get_best(npz_files, n, recursive, metadata_field)
        else:
            filename = f
            fit = gc_datafile.load(f)
            score = fit['metadata'][metadata_field]
            fits = [(score, filename)]
        sorted_fits = sorted(sorted_fits + fits, key=lambda x: x[0])[:n]
    return sorted_fits

def main():
    parser = argparse.ArgumentParser(description = 'Choose the best fit according to their final optimization score')
    parser.add_argument('fit', help='.npz file or directory of them, created by gc_model', type=str, nargs='+')
    parser.add_argument('-t', '--target', help='Symlink the best fits to this target. {n} will be replaced with rank.', type=str, required=False)
    parser.add_argument('--no-recursive', help='Ignore directories instead of recursing into them', dest='recursive', action='store_false', required=False)
    parser.add_argument('-n', help='Top N to show. Defaults to 1.', type=int, required=False, default=1)
    args = parser.parse_args()

    fits = get_best(args.fit, n=args.n, recursive=args.recursive)

    print(f"Top {args.n} fits:")
    for _, filename in fits:
        print(filename)

    if args.target:
        for n, (_, filename) in enumerate(fits):
            parsed_target = args.target.format(n=n+1)
            if os.path.exists(parsed_target):
                print(f"Fit {n+1} target already exists, skipping:", parsed_target)
            else:
                os.symlink(os.path.abspath(filename), parsed_target)
                print(f"Fit {n+1} linked to target:", parsed_target)

if __name__ == '__main__':
    main()
