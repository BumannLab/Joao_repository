import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IN_FNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/pixels_intensity_over1000.tsv"
ODIRNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/Images/"
BINS = 650
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
    xmin, xman = 0, 65535
    ymin, ymax = 0, 65535

    xedges = np.linspace(xmin, xman, num_bins + 1)
    yedges = np.linspace(ymin, ymax, num_bins + 1)
    
    return xedges, yedges


def combine_cols_to_plot(columns_to_plot, column_of_interest):
    return [(column_of_interest, col) for col in columns_to_plot if col != column_of_interest]


def rename_columns(old_key, old_new_keys):
    new_key = []
    for channel in old_key.split("-"):    
        new_key.append(old_new_keys[channel])
    return "/".join(new_key)


def plot_scatters(odirname, counts, columns_combn, xedges, yedges, channel_map):

    cmap = plt.cm.magma.copy()
    cmap.set_under("lightgray")
    
    for combn in columns_combn:        
        col1, col2 = combn

        col1 = rename_columns(col1, channel_map)
        col2 = rename_columns(col2, channel_map)

        plt.figure(figsize=(12, 8))
        plt.imshow(counts["+".join(combn)], cmap=cmap, vmin=1, interpolation='nearest', origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
                )
        
        plt.xticks(np.arange(0, 65536, 2000), rotation='vertical')
        plt.yticks(np.arange(0, 65536, 2000))

        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)

        plt.colorbar(label="count")
        plt.grid(alpha=0.2, linestyle="--")

        plt.tight_layout()

        plt.savefig(f"{odirname}/{col1}_vs_{col2}_scatter2d_test.pdf", dpi=300, bbox_inches="tight")


def main(input_file, odirname, bins, channel_map, column_of_interest):
    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    chunk_size = 50_000_000

    header = get_header(input_file)
    columns_to_plot = get_columns_to_plot(header)

    cols_combn = combine_cols_to_plot(columns_to_plot, column_of_interest)

    xedges, yedges = obtain_bin_info(bins)

    chunked_df = pd.read_csv(input_file, sep="\t", usecols=columns_to_plot, dtype="float32",
                        chunksize=chunk_size# , nrows=chunk_size
                        )

    counts = {"+".join(combn): np.zeros((bins, bins), dtype=np.int32) for combn in cols_combn}
   
    for chunk in chunked_df:
        for col1, col2 in cols_combn:
            column1 = chunk[col1].to_numpy(copy=False)
            column2 = chunk[col2].to_numpy(copy=False)
            bin_distribution, _, _ = np.histogram2d(column1, column2, bins=(xedges, yedges))
            counts[f"{col1}+{col2}"] = bin_distribution.T

    plot_scatters(odirname, counts, cols_combn, xedges, yedges, channel_map)


if __name__ == "__main__":
    start_time = time.time()
    main(IN_FNAME, ODIRNAME, BINS, CHANNEL_MAP, COLUMN_OF_INTEREST)
    print(f"Finished in {time.time() - start_time}")
 