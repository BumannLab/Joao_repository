import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IN_FNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/pixels_intensity_over1000.tsv"
ODIRNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/Images/"
BINS = 100
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
    columns_to_plot = [col for col in all_columns if re.search("/", col)]
    return columns_to_plot


def obtain_bin_info(num_bins):
    xmin = 0
    xmax = 3
    edges = np.linspace(xmin, xmax, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths  = np.diff(edges)
    return edges, centers, widths


def combine_counts(counts, bins, columns_to_plot):
    final_counts = {c: np.zeros(bins, dtype=np.int64) for c in columns_to_plot}
    for count in counts:
        for col in columns_to_plot:
            final_counts[col] += count[col]
    return final_counts


def rename_columns(old_key, old_new_keys):
    new_key = []
    for channel in old_key.split("/"):    
        new_key.append(old_new_keys[channel])
    return "/".join(new_key)


def plot_histogram(odirname, counts, columns_to_plot, centers, channel_map):
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(columns_to_plot)))

    for c, color in zip(columns_to_plot, colors):        
        
        plt.fill_between(centers, counts[c], step="mid", alpha=0.3, color=color)
        plt.step(centers, counts[c], where="mid", color=color, linewidth=2, label=rename_columns(c, channel_map))

        #or this, but use edges not centers !!
        # plt.hist(edges[:-1], edges, weights=counts[c], label=rename_columns(c, channel_map), density=True, color=color, alpha = 0.7, edgecolor="black", linewidth=0.7)

    plt.xlabel("Ratio", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{odirname}/Ratios_distribution_histogram.pdf", dpi=300, bbox_inches="tight")


def main(input_file, odirname, bins, channel_map):
    chunk_size = 50_000_000

    header = get_header(input_file)
    columns_to_plot = get_columns_to_plot(header)

    edges, centers, widths = obtain_bin_info(bins)
    
    chunked_df = pd.read_csv(input_file, sep="\t", usecols=columns_to_plot, dtype="float32",
                        chunksize=chunk_size)

    counts = {c: np.zeros(bins, dtype=np.int32) for c in columns_to_plot}
   
    for chunk in chunked_df:
        for col in columns_to_plot:
            column = chunk[col].to_numpy(copy=False)
            bin_distribution, _ = np.histogram(column, bins=edges)
            counts[col] += bin_distribution

    plot_histogram(odirname, counts, columns_to_plot, centers, channel_map)


if __name__ == "__main__":
    start_time = time.time()
    main(IN_FNAME, ODIRNAME, BINS, CHANNEL_MAP)
    print(f"Finished in {time.time() - start_time}")
 