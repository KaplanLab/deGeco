import pandas
import numpy as np
import os
import multiprocessing
import functools
import warnings
import glob

chr_lengths = np.array([4980, 4844, 3966, 3805, 3631, 3417, 3187, 2903, 2768, 2676, 2702, 2666, 2288, 2141, 2040, 1807, 1666, 1608, 1173, 1289, 935, 1017])

def get_coords_tab(filename, fit, res=50000):
    coords = pandas.read_csv(filename, sep='\t')
    coords[['chromosome', 'start', 'end']] = coords['Genomic coordinate'].str.split(pat='[:-]', expand=True)
    coords[['start', 'end']] = coords[['start', 'end']].astype(int)
    coords['mid'] = (coords['start'] + coords['end']) // 2
    coords[['start_bin', 'end_bin', 'mid_bin']] = coords[['start', 'end', 'mid']] // res

    lambda_columns = [f'lambda{i+1}' for i in range(fit['lambdas'].shape[1])]
    coords[lambda_columns] = pandas.DataFrame(fit['lambdas'][coords['mid_bin']])
    
    return coords

def pairwise_distance(tab, bin_count):
    N = len(tab)
    distances = np.full((bin_count, bin_count), np.nan)
    coords = tab[['X(nm)', 'Y(nm)', 'Z(nm)']].values
    distances_unordered = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    bins_order = tab['mid_bin'].values
    for i, x in enumerate(bins_order):
        for j, y in enumerate(bins_order):
            distances[x, y] = distances_unordered[i, j]
    return distances

def calc_separation(coords):
    coords_columns = ['X(nm)', 'Y(nm)', 'Z(nm)']
    strong_lambda1 = coords[coords['lambda1'] > 0.75]
    strong_lambda2 = coords[coords['lambda1'] < 0.24]
    mixed_lambda = coords[(0.23 < coords['lambda1']) & (coords['lambda1'] < 0.74)]

    strong_lambda1_center = np.nanmean(strong_lambda1[coords_columns], axis=0)
    strong_lambda2_center = np.nanmean(strong_lambda2[coords_columns], axis=0)
    mixed_lambda_center = np.nanmean(mixed_lambda[coords_columns], axis=0)

    strong_lambda1_radius = np.nanquantile(np.linalg.norm(strong_lambda1[coords_columns] - strong_lambda1_center, axis=1), 0.5)
    strong_lambda2_radius = np.nanquantile(np.linalg.norm(strong_lambda2[coords_columns] - strong_lambda2_center, axis=1), 0.5)
    mixed_lambda_radius = np.nanquantile(np.linalg.norm(mixed_lambda[coords_columns] - mixed_lambda_center, axis=1), 0.5)

    l2_l1 = np.nanmean(np.linalg.norm(strong_lambda2[coords_columns] - strong_lambda1_center, axis=1) < strong_lambda1_radius)
    l1_l2 = np.nanmean(np.linalg.norm(strong_lambda1[coords_columns] - strong_lambda2_center, axis=1) < strong_lambda2_radius)
    mixed_l1 = np.nanmean(np.linalg.norm(mixed_lambda[coords_columns] - strong_lambda1_center, axis=1) < strong_lambda1_radius)
    mixed_l2 = np.nanmean(np.linalg.norm(mixed_lambda[coords_columns] - strong_lambda2_center, axis=1) < strong_lambda2_radius)
    l1_mixed = np.nanmean(np.linalg.norm(strong_lambda1[coords_columns] - mixed_lambda_center, axis=1) < mixed_lambda_radius)
    l2_mixed = np.nanmean(np.linalg.norm(strong_lambda2[coords_columns] - mixed_lambda_center, axis=1) < mixed_lambda_radius)
    
    return l1_l2, l2_l1, mixed_l1, mixed_l2, l1_mixed, l2_mixed, strong_lambda1_center, strong_lambda2_center, mixed_lambda_center

def process(coords_all, instance):
    print(instance, ": Running")

    distances_output = f'chr21_distances/chr21_distances_i{instance}.npz.npy'
    pct_output = f'chr21_pct/chr21_pct_i{instance}.npz'

    coords = coords_all[coords_all['Chromosome copy number'] == instance]

    if os.path.exists(distances_output):
        print(f"Skipping distance calculation: {distances_output} exists")
    else:
        distances = pairwise_distance(coords, bin_count=chr_lengths[20])
        np.save(distances_output, distances)

    if os.path.exists(pct_output):
        print(f"Skipping pct calculation: {pct_output} exists")
    else:
        l1_l2, l2_l1, mixed_l1, mixed_l2, l1_mixed, l2_mixed, l1_center, l2_center, mixed_center = calc_separation(coords)
        np.savez(pct_output, l1_l2=l1_l2, l2_l1=l2_l1, mixed_l1=mixed_l1, mixed_l2=mixed_l2, l1_mixed=l1_mixed,
                l2_mixed=l2_mixed, l1_center=l1_center, l2_center=l2_center, mixed_center=mixed_center)

    print(instance, ": Done", distances_output, pct_output)
    return (distances_output, pct_output)

def main():
    imr90_2s_fit = dict(np.load('imr90_chr21_2states_fit.npz'))
    coords_chr21 = get_coords_tab('3d_coords_chr21.tsv', imr90_2s_fit)
    instances = coords_chr21['Chromosome copy number'].max()

    process_instance = functools.partial(process, coords_chr21)
    proc_count = os.cpu_count()
    chunksize = int(np.ceil(instances / proc_count))
    p = multiprocessing.Pool(proc_count)
    print(f"*** Starting job with {proc_count} processes ***")
    with p as pool:
       output_files = p.imap_unordered(process_instance, range(1, instances+1), chunksize=chunksize)
       for o in output_files:
           print(o)

def create_bulk_hic(distances_dir, threshold=500):
    g = glob.glob(f'{distances_dir}/chr21_distances_*')
    counter = np.zeros(np.load(g[0]).shape)
    hic = np.zeros(np.load(g[0]).shape)
    for i, f in enumerate(g):
        print("Instance", i)
        d = np.load(f)
        nn = np.isfinite(d)
        counter += nn
        hic_cell = d < threshold
        hic += hic_cell

    return hic, counter

if __name__ == '__main__':
    main()
