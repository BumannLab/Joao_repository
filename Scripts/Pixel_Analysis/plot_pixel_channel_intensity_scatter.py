import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IN_FNAME = (
    "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/pixels_intensity_over1000.tsv"
)
ODIRNAME = "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/Plots"
COLUMN_OF_INTEREST = "SYBRGold"
BINS = 655
LOG_SCALE = True


def get_header(file: Path):
    with open(file, 'r') as infile:
        all_columns = infile.readline().rstrip().split("\t")
    return all_columns


def get_columns_to_plot(all_columns: list[str]):
    columns_to_plot = [col for col in all_columns if "intensity" in col.lower() and "/" not in col]
    return columns_to_plot


def combine_cols_to_plot(columns_to_plot, column_of_interest):
    return [(column_of_interest + "_Intensity", col) for col in columns_to_plot if column_of_interest.lower() not in col.lower()]


def obtain_bin_info(num_bins, log_scale):
    xmin, xmax = 0, 65535
    ymin, ymax = 0, 65535
    
    if log_scale:
        xedges = 10 ** np.linspace(xmin, np.log10(xmax), num_bins + 1)
        yedges = 10 ** np.linspace(ymin, np.log10(ymax), num_bins + 1)
    else:
        xedges = np.linspace(xmin, xmax, num_bins + 1)
        yedges = np.linspace(ymin, ymax, num_bins + 1)

    return xedges, yedges


def find_max_x_y_edge(counts):

    mask = counts != 0

    cols_with_data = np.where(mask.any(axis=0))[0]
    rows_with_data = np.where(mask.any(axis=1))[0]

    max_x_edge = cols_with_data.max() + 1
    max_y_edge = rows_with_data.max() + 1
    
    return max_x_edge, max_y_edge


def plot_scatters(odirname, counts, columns_combn, xedges, yedges, logscale = False):

    cmap = plt.cm.magma.copy()
    cmap.set_under("lightgray")
    
    for combn in columns_combn:        
        col1, col2 = combn

        x_label = col1.replace("_Intensity", "")
        y_label = col2.replace("_Intensity", "")
        out_fname = f"{odirname}/{x_label}_vs_{y_label}_pixel_intensity_scatter2d_test_small_nrows.pdf"

        combn_counts = counts["+".join(combn)]

        xedge_max, yedge_max = find_max_x_y_edge(combn_counts)

        plt.figure(figsize=(12, 8))
        plt.imshow(combn_counts[:yedge_max, :xedge_max], cmap=cmap, vmin=1, interpolation='nearest', aspect='auto',
                    origin='lower', extent=[xedges[0], xedges[xedge_max], yedges[0], yedges[yedge_max]]
                ) 

        if logscale:

            x_label = f"log10({x_label})"
            y_label = f"log10({y_label})"
            out_fname = out_fname.replace(".pdf", "_log10.pdf")
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        plt.colorbar(label="count")
        plt.grid(alpha=0.2, linestyle="--")

        plt.tight_layout()

        plt.savefig(out_fname, dpi=300, bbox_inches="tight")


def main(input_file, odirname, bins, column_of_interest, log_scale):
    
    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    chunk_size = 1_000

    header = get_header(input_file)
    columns_to_plot = get_columns_to_plot(header)

    cols_combn = combine_cols_to_plot(columns_to_plot, column_of_interest)

    xedges, yedges = obtain_bin_info(bins, log_scale)

    chunked_df = pd.read_csv(input_file, sep="\t", usecols=columns_to_plot, dtype="int32",
                        chunksize = chunk_size, nrows=chunk_size * 3)

    counts = {"+".join(combn): np.zeros((bins, bins), dtype=np.float32) for combn in cols_combn}
 
    for chunk in chunked_df:
        
        for col1, col2 in cols_combn:
            column1 = chunk[col1].to_numpy(copy=False)
            column2 = chunk[col2].to_numpy(copy=False)
            
            bin_distribution, _, _ = np.histogram2d(column1, column2, bins=(xedges, yedges))
            
            counts[f"{col1}+{col2}"] += bin_distribution.T

    plot_scatters(odirname, counts, cols_combn, xedges, yedges, log_scale)


if __name__ == "__main__":
    start_time = time.time()
    main(IN_FNAME, ODIRNAME, BINS, COLUMN_OF_INTEREST, LOG_SCALE)
    print(f"Finished in {time.time() - start_time}")
 