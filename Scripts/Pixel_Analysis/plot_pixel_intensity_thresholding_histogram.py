import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IN_FNAME = (
    "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/pixels_intensity_over1000.tsv"
    "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/pixels_intensity_over1000.tsv"
)
ODIRNAME = "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/Plots/"
BINS = 100


def get_header(file: Path):
    with open(file, "r") as infile:
        all_columns = infile.readline().rstrip().split("\t")
    return all_columns


def get_columns_to_plot(all_columns: list[str]):
    columns_to_plot = [col for col in all_columns if "/" in col]
    return columns_to_plot


def obtain_bin_info(num_bins):
    xmin = 0
    xmax = 3
    edges = np.linspace(xmin, xmax, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    return edges, centers, widths


def plot_histogram(odirname, counts, columns_to_plot, centers):
    plt.figure(figsize=(12, 8))

    colors = [
        "#377eb8",
        "#ff7f00",
        "#f781bf",
        "#a65628",
        "#4daf4a",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    for c, color in zip(columns_to_plot, colors):

        plt.fill_between(centers, counts[c], step="mid", alpha=0.3, color=color)
        plt.step(centers, counts[c], where="mid", color=color, linewidth=2, label = c)

        # plt.hist(
        #     edges[:-1],
        #     edges,
        #     weights=counts[c],
        #     label=rename_columns(c, channel_map),
        #     color=color,
        #     alpha=0.5,
        #     edgecolor="black",
        #     linewidth=0.7,
        # )

    plt.xlabel("Ratio", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(
        f"{odirname}/Ratios_distribution_histogram_step.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def main(input_file, odirname, bins):

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    chunk_size = 50_000_000

    header = get_header(input_file)
    columns_to_plot = get_columns_to_plot(header)

    edges, centers, widths = obtain_bin_info(bins)

    chunked_df = pd.read_csv(
        input_file,
        sep="\t",
        usecols=columns_to_plot,
        dtype="float32",
        chunksize=chunk_size
    )

    counts = {c: np.zeros(bins, dtype=np.int32) for c in columns_to_plot}

    for chunk in chunked_df:
        for col in columns_to_plot:
            column = chunk[col].to_numpy(copy=False)
            bin_distribution, _ = np.histogram(column, bins=edges)
            counts[col] += bin_distribution

    plot_histogram(odirname, counts, columns_to_plot, centers)


if __name__ == "__main__":
    start_time = time.time()
    main(IN_FNAME, ODIRNAME, BINS)
    print(f"Finished in {time.time() - start_time}")
