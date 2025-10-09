import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IN_FNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/pixels_intensity_over1000.tsv"
OUT_FNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/Images/Ratios_distribution_histogram_test_hist_with_line.pdf"
BINS = 50
COLUMN_OF_INTEREST = "Channel_2"
CHANNEL_MAP={
    "Channel_1":"AL-555",
    "Channel_2":"SYBR_GOLD",
    "Channel_3":"DAPI",
    "Channel_4":"AL-647",
}


def get_header(file: Path):
    with open(file, 'r') as infile:
        all_columns = infile.readline().rstrip().split("\t")
    return all_columns


def get_columns_to_plot(all_columns: list[str]):
    columns_to_plot = [col for col in all_columns if re.search("^Channel_[0-9]$", col)]
    return columns_to_plot


def obtain_bin_info(num_bins):
    xedges = np.linspace(0, 65535, num_bins + 1)
    yedges = np.linspace(0, 65535, num_bins + 1)
    
    centers = (xedges[:-1] + xedges[1:]) / 2

    widths  = np.diff(yedges)
    
    return xedges, yedges, centers, widths


def combine_cols_to_plot(columns_to_plot, column_of_interest):
    return [(column_of_interest, col) for col in columns_to_plot if col != column_of_interest]


def rename_columns(old_key, old_new_keys):
    new_key = []
    for channel in old_key.split("_"):    
        new_key.append(old_new_keys[channel])
    return "/".join(new_key)


def plot_histogram(out_fname, counts, columns_to_plot, centers, channel_map):
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(columns_to_plot)))

    for c, color in zip(columns_to_plot, colors):        
        
        plt.fill_between(centers, counts[c], step="mid", alpha=0.3, color=color)
        plt.step(centers, counts[c], where="mid", color=color, linewidth=2, label=rename_columns(c, channel_map))

        #or this, but use edges not centers ()
        # plt.hist(edges[:-1], edges, weights=counts[c], label=rename_columns(c, channel_map), density=True, color=color, alpha = 0.7, edgecolor="black", linewidth=0.7)


    plt.xlabel("Ratio", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_fname, dpi=300, bbox_inches="tight")


def main(input_file, out_file, bins, channel_map, column_of_interest):
    chunk_size = 50_000

    header = get_header(input_file)
    columns_to_plot = get_columns_to_plot(header)
    cols_combn = combine_cols_to_plot(columns_to_plot, column_of_interest)

    xedges, yedges, centers, widths = obtain_bin_info(bins)

    chunked_df = pd.read_csv(input_file, sep="\t", usecols=columns_to_plot, dtype="float32",
                        chunksize=chunk_size, nrows = chunk_size)

    counts = {"_".join(combn): np.zeros((bins, bins), dtype=np.int32) for combn in cols_combn}
   
    for chunk in chunked_df:
        print(chunk)
        for col1, col2 in cols_combn:
            column1 = chunk[col1].to_numpy(copy=False)
            column2 = chunk[col2].to_numpy(copy=False)
            bin_distribution, _, _ = np.histogram2d(column1, column2, bins=(xedges, yedges))
            counts[f"{col1}_{col2}"] = bin_distribution.T

    fig = plt.figure(figsize=(7, 3))

    ax = fig.add_subplot(131, title='imshow: square bins')

    plt.imshow(counts["Channel_2_Channel_1"], interpolation='nearest', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    ax = fig.add_subplot(132, title='pcolormesh: actual edges',

        aspect='equal')

    X, Y = np.meshgrid(xedges, yedges)

    ax.pcolormesh(X, Y, counts["Channel_2_Channel_1"])


    
    plt.show()

    return
    plot_histogram(out_file, counts, columns_to_plot, centers, channel_map)


if __name__ == "__main__":
    start_time = time.time()
    main(IN_FNAME, OUT_FNAME, BINS, CHANNEL_MAP, COLUMN_OF_INTEREST)
    print(f"Finished in {time.time() - start_time}")
 