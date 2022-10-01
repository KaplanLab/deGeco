import sys
import traceback
import itertools

import clodius.tiles.format as hgfo
import numpy as np
import gc_datafile

fit = FitMatrix(gc_datafile.load_params('output/all_no_ym_stretched_3st_50000_z406400091_t5.npz'))

def cis_trans_mask_slice(cis_lengths, slice_x, slice_y):
    groups = np.arange(np.size(cis_lengths))
    groups_per_bin = np.repeat(groups, cis_lengths)
    groups_per_bin_x = groups_per_bin[slice_x]
    groups_per_bin_y = groups_per_bin[slice_y]
    return groups_per_bin_x[:, None] == groups_per_bin_y[None, :]

def log_gc_interactions_slice(lambdas, cis_weights, trans_weights, cis_trans_mask, slice_x,
                              slice_y):
    lambdas_x = lambdas[slice_x]
    lambdas_y = lambdas[slice_y]
    cis_interactions = lambdas_x @ cis_weights @ lambdas_y.T
    trans_interactions = lambdas_x @ trans_weights @ lambdas_y.T
    combined_interactions = np.where(cis_trans_mask, cis_interactions, trans_interactions)

    return np.log(combined_interactions)

def log_dd_interactions_slice(alpha, beta, n, cis_trans_mask, slice_x, slice_y):
    bin_distances = np.arange(n)
    bin_distances_x = bin_distances[slice_x]
    bin_distances_y = bin_distances[slice_y]
    distances = 1.0 * np.abs(bin_distances_x[:, None] - bin_distances_y[None, :])
    distances[distances == 0] = np.nan # Remove main diag
    cis_interactions = alpha * np.log(distances)
    trans_interactions = np.full_like(cis_interactions, beta)
    
    return np.where(cis_trans_mask, cis_interactions, trans_interactions)

def generate_interactions_mat_slice(slice_x, slice_y, state_probabilities, cis_weights,
                                    trans_weights, cis_dd_power, trans_dd, cis_lengths):     
    cis_trans_mask = cis_trans_mask_slice(cis_lengths, slice_x, slice_y)
    gc_interaction = log_gc_interactions_slice(state_probabilities, cis_weights, trans_weights,
                                               cis_trans_mask, slice_x, slice_y)
    dd_interaction = log_dd_interactions_slice(cis_dd_power, trans_dd, np.sum(cis_lengths),
                                               cis_trans_mask, slice_x, slice_y) 
    matrix =  np.exp(gc_interaction + dd_interaction)
    
    return matrix

class FitMatrix:
    def __init__(self, fit):
        self._fit = fit
        for key in ('state_probabilities', 'cis_weights', 'trans_weights', 'trans_dd', 'cis_dd_power', 'cis_lengths'):
            setattr(self, key, fit[key])
        n = np.sum(fit['cis_lengths'])
        self.chr_offsets = np.concatenate([[0], np.cumsum(fit['cis_lengths'])])
        self.shape = (n, n)
        self._binning_factor = 1
        
    def bin_by(self, factor):
        new_cis_lengths = np.ceil(np.array(self.cis_lengths) / factor).astype(int)
        per_chr_lambdas = np.split(self.state_probabilities, self.chr_offsets[1:-1])
        new_per_chr_lambdas = []
        for new_length, chr_lambda in zip(new_cis_lengths, per_chr_lambdas):
            rem = new_length * factor - chr_lambda.shape[0]
            padded_lambda = np.pad(chr_lambda, [(0, rem), (0, 0)], constant_values=np.nan)
            binned_lambda = np.nanmean(np.reshape(padded_lambda, (new_length, factor, -1)), axis=1)
            new_per_chr_lambdas.append(binned_lambda)
        new_lambdas = np.concatenate(new_per_chr_lambdas)
        new_fit = {**self._fit, 'state_probabilities': new_lambdas, 'cis_lengths': new_cis_lengths}
        new_obj = FitMatrix(new_fit)
        new_obj._binning_factor = factor
        
        return new_obj
    
    def __getitem__(self, key):
        try:
            x, y = key
        except:
            raise ValueError("Must use 2-d indexing")
        if not isinstance(x, slice):
            x = slice(x, x+1)
        if not isinstance(y, slice):
            y = slice(y, y+1)
        return generate_interactions_mat_slice(x, y, **self._fit)

resolutions = tuple(reversed((
  50000,
  100000,
  250000,
  500000,
  1000000,
  2000000,
  4000000,
  8000000
)))

def tileset_info():
    return {'resolutions': resolutions,
 'max_pos': [3095693983, 3095693983],
 'min_pos': [1, 1],
 #'chromsizes': [['chr1', 249250621],
 # ['chr2', 243199373],
 # ['chr3', 198022430],
 # ['chr4', 191154276],
 # ['chr5', 180915260],
 # ['chr6', 171115067],
 # ['chr7', 159138663],
 # ['chr8', 146364022],
 # ['chr9', 141213431],
 # ['chr10', 135534747],
 # ['chr11', 135006516],
 # ['chr12', 133851895],
 # ['chr13', 115169878],
 # ['chr14', 107349540],
 # ['chr15', 102531392],
 # ['chr16', 90354753],
 # ['chr17', 81195210],
 # ['chr18', 78077248],
 # ['chr19', 59128983],
 # ['chr20', 63025520],
 # ['chr21', 48129895],
 # ['chr22', 51304566],
 # ['chrX', 155270560],
 # ['chrY', 59373566],
 # ['chrM', 16571]]
           }
def tile_raw(tile_id):
    BINS_PER_TILE = 256
    _, zoom, x_pos, y_pos = tile_id.split('.')
    zoom = int(zoom)
    x_pos = int(x_pos)
    y_pos = int(y_pos)
    resolution = resolutions[zoom]
    scale_factor = int(np.ceil(resolution/resolutions[-1]))

    start1 = x_pos * BINS_PER_TILE
    end1 = (x_pos + 1) * BINS_PER_TILE
    start2 = y_pos * BINS_PER_TILE
    end2 = (y_pos + 1) * BINS_PER_TILE
    binned_fit = fit.bin_by(scale_factor)
    try:
        tile_data = binned_fit[start2:end2, start1:end1]
        tile_data = np.pad(tile_data, [(0, BINS_PER_TILE - tile_data.shape[0]), (0, BINS_PER_TILE - tile_data.shape[1])])

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    return tile_data

def tiles(tile_ids):

    tiles = []
    for tile_id in tile_ids:
        tile_data = tile_raw(tile_id)
        tiles.append((tile_id, hgfo.format_dense_tile(tile_data)))
        
    return tiles   
