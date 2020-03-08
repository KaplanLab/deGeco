import sys
import os
import argparse
import re
import glob
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt, gridspec

import hic_analysis as hic
import gc_model

def all_figures(input_data, fit1, fit2, histones_dir):
    figure1 = hic_figure(input_data, *fit1)
    figure2 = hic_figure(input_data, *fit2, figure1.axes[0])
    figure3 = histones_figure(histones_dir, x_axis=figure1.axes[0])
    #figure4 = probabilities_histogram_figure(l)
    plt.show()

def hic_figure(input_data, l, weights, alpha, x_axis=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=4, ncols=4, figure=fig)

    ax1 = fig.add_subplot(gs[:3, :3], sharex=x_axis, sharey=x_axis)
    ax1.set_aspect('equal', 'box')
    plot_data(ax1, input_data, l, alpha, weights)

    ax2 = fig.add_subplot(gs[3, :3], sharex=x_axis or ax1)
    plot_probabilities(fig, ax2, l)

    #ax3 = fig.add_subplot(gs[0, 2])
    #plot_weights(ax3, weights)

    return fig

def histones_figure(histones_dir, x_axis=None, ylim_min_pct=0, ylim_max_pct=99):
    fig = plt.figure()
    histones = glob.glob(f'{histones_dir}/*')
    histones_count = len(histones)
    for i, histone_filename in enumerate(histones):
        ax = fig.add_subplot(histones_count, 1, i+1, sharex=x_axis)
        plot_histone(fig, ax, histone_filename, ylim_max_pct=ylim_max_pct, ylim_min_pct=ylim_min_pct)

def probabilities_histogram_figure(l):
    n_states = l.shape[1]
    fig = plt.figure()
    for state in range(n_states):
        ax = fig.add_subplot(n_states, 1, state+1)
        probabilities = hic.remove_nan(l[:, state])
        ax.hist(probabilities)
        ax.set_title(f'Probabilities histogram for state {state+1}')

    return fig

def set_hidable_lines(figure, ax, legend, lines):
    lined = dict()
    for legline, origline in zip(legend.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined.get(legline)
        if not origline:
            return
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        figure.canvas.draw()

    figure.canvas.mpl_connect('pick_event', onpick)

def plot_histone(fig, ax, histone_filename, ylim_min_pct=0, ylim_max_pct=99):
    histone_mod_title = get_histone_modification_title(histone_filename)
    histone_mod = np.load(histone_filename)
    ylim_min = np.percentile(histone_mod, ylim_min_pct)
    ylim_max = np.percentile(histone_mod, ylim_max_pct)
    ax.set_title(f'Histone mod: {histone_mod_title}')
    ax_lines = ax.plot(histone_mod)
    ax.set_ylim(bottom=ylim_min, top=ylim_max)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_legend = ax.legend(['min_with_zeros', 'min', 'max'], bbox_to_anchor=(1, 1), loc='upper left')
    set_hidable_lines(fig, ax, ax_legend, ax_lines)
    
def plot_weights(ax, weights):
    ax.set_title('weights')
    ax_img = ax.imshow(weights)
    plt.colorbar(ax_img)

def plot_probabilities(fig, ax, l):
    ax.set_title('probabilities vectors for each compartment')
    ax_lines = ax.plot(l) 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_legend = ax.legend([f'c{i+1}' for i, _ in enumerate(ax_lines)], bbox_to_anchor=(1, 1), loc='upper left')
    set_hidable_lines(fig, ax, ax_legend, ax_lines)

def plot_data(ax, input_data, l, alpha, weights):
    data = hic.preprocess(input_data)
    data_reconstruction = gc_model.generate_interactions_matrix(l, weights, alpha)
    log_normalized = lambda x: hic.safe_log(hic.normalize_distance(x))

    data_and_model = hic.merge_by_diagonal(log_normalized(data), log_normalized(data_reconstruction))
    states = weights.shape[0]
    ax.set_title(f'states={states}')
    ax_img = ax.imshow(data_and_model, aspect='equal')
    plt.colorbar(ax_img)

def get_histone_modification_title(filename):
    basename = os.path.basename(filename)
    match = re.search('(H[0-9][a-z][0-9]+[a-z]+[0-9]*)', basename)

    if match:
        return match[1]

def detect_file_type(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.npy', '.mcool']:
        return ext[1:]
    raise ValueError(f'Unknown file type "{ext}" from filename: {filename}')

def verify_shapes(input_data, l, alpha, weights):
    assert input_data.shape[0] == input_data.shape[1], 'Input data must be a square matrix'
    assert weights.shape[0] == weights.shape[1], 'weights must be a square matrix'

    number_of_bins = input_data.shape[0]
    number_of_states = weights.shape[0]
    
    assert l.shape == (number_of_bins, number_of_states), 'lambdas must be of shape BINSxSTATES'
    assert np.ndim(alpha) == 0, 'alpha must be a scalar'

def main():
    parser = argparse.ArgumentParser(description = 'Visualize results from model')
    parser.add_argument('-d', help='experiment data in mcool or npy format', dest='data', type=str, required=True)
    parser.add_argument('-t', help='file type of experiment data file', dest='type', type=str,
            choices=['mcool', 'npy', 'auto'], default='auto')
    parser.add_argument('-c', help='chromosome number (required for type mcool)', dest='chrom', type=int)
    parser.add_argument('-r', help='resolution (required for type mcool)', dest='resolution', type=int)
    parser.add_argument('-l', help='file containing probability parameters', dest='l', type=str, required=True)
    parser.add_argument('-a', help='file containing distance decay power value', dest='alpha', type=str, required=True)
    parser.add_argument('-w', help='file containing weights matrix', dest='weights', type=str, required=True)
    args = parser.parse_args()

    file_type = detect_file_type(args.data) if args.type == 'auto' else args.type
    if file_type == 'mcool':
        if args.chrom is None or args.resolution is None:
            print("chromosome and resolution must be given for mcool files")
            sys.exit(1)
        chrom_name = f'chr{args.chrom}'
        input_data = get_matrix_from_coolfile(args.data, args.resolution, chrom_name)
    else:
        input_data = np.load(args.data)

    l = np.load(args.l)
    alpha = np.load(args.alpha)
    weights = np.load(args.weights)

    verify_shapes(input_data, l, alpha, weights)
    all_figures(input_data, l, alpha, weights, './histone_modifications')
    
if __name__ == '__main__':
    main()
