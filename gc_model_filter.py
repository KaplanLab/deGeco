#!/usr/bin/python3
import argparse
import glob
import os
import numpy as np

import gc_model

def get_best(files):
    best_filename = None
    best_score = np.inf
    for f in files:
        if os.path.isdir(f):
            npz_files = glob.glob(f'{f}/*.npz')
            score, filename = get_best(npz_files)
        else:
            filename = f
            fit = np.load(f, allow_pickle=True)
            score = fit['metadata'][()]['optimize_value']
        if score < best_score:
            best_score = score
            best_filename = filename
    return best_score, best_filename

def main():
    parser = argparse.ArgumentParser(description = 'Choose the best fit according to their final optimization score')
    parser.add_argument('fit', help='.npz file or directory of them, created by gc_model', type=str, nargs='+')
    parser.add_argument('--target', help='If given, will symlink the best file to this target', type=str, required=False)
    args = parser.parse_args()

    score, filename = get_best(args.fit)

    print(f"Best fit is:")
    print(filename)

    if args.target:
        os.symlink(os.path.abspath(filename), args.target)
        print("Fit linked to target", args.target)

if __name__ == '__main__':
    main()
