import csv
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import random

from MCrop_events_gui.gui import EventBrowserGui
from utils import read_image, get_images_path

CLUSTER_POSITIONS_FILE = "Results/B.e2024.21394/SYBRGold_CD68AL555_LAMAL647_DAPI-2M_ammonium_acetate/SYBRGold_GET800/SYBRGold_Mean_GET1000-DAPI_Mean_LET250/voxel_clusters_size_over5_SYBRGold_greater_equal_800.tsv"
IMAGES_DIR = "/data/joao/mTBData/20251127_eB.2024.21394/full_res_2M_ammonium_acetate/Sybrgold_cd68_lam_dapi"

NUMBER_EVENTS = "all" # Integer e.g. 2000 or "all" e.g. will take all the file positions
EVENT_WIDTH = 256
EVENT_HEIGHT = 256
DISPLAY_CHANNELS = (0,1,3)
FIJI_EXECUTABLE = "/data/joao/Fiji.app/ImageJ-linux64" # e.g. "/path/to/Fiji.app/ImageJ-linux64"

ODIRNAME = "NN_Dataset/RAW_DATASET/B.e202.21394/UNet bad prediction"

@dataclass
class Cluster:
    tile_name: str
    x_position: int
    y_position: int


def read_cluster_data(file_path):
    try:
        import polars as pl
    except ImportError:
        return read_cluster_data_with_csv(file_path)

    cluster_rows = (
        pl.scan_csv(file_path, separator="\t")
        .filter(pl.col("Volume").cast(pl.Utf8) != "-1")
        .select("Image", "X", "Y")
        .collect()
    )

    cluster_data = [
        Cluster(row["Image"], float(row["X"]), float(row["Y"]))
        for row in cluster_rows.iter_rows(named=True)
    ]

    return cluster_data


def read_cluster_data_with_csv(file_path):
    cluster_data = list()

    with open(file_path, mode="r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")

        for row in reader:
            if row["Volume"] != "-1": # Filter rows of tiles with no clusters
                cluster = Cluster(row["Image"], float(row["X"]), float(row["Y"]))
                cluster_data.append(cluster)

    return cluster_data


def select_subsample(cluster_data, num_events):

    data_size = len(cluster_data)
    
    if not isinstance(num_events, str):
        assert int(num_events) <= data_size

    if num_events == "all":
        num_events = data_size
        sample = cluster_data

    else:
        sample = random.sample(cluster_data, k = num_events)

    sample_dict = defaultdict(list)

    for data in sample:
        sample_dict[data.tile_name].append(data)
    return sample_dict


def get_cutting_position(image_height_width, cluster, event_height, event_width):
    image_height, image_width = image_height_width

    if event_width > image_width or event_height > image_height:
        raise ValueError(
            "Event crop cannot be larger than the source image. "
            f"Image shape: {image_height_width}; event shape: {(event_height, event_width)}."
        )

    xmin = int(round(cluster.x_position - (event_width / 2)))
    ymin = int(round(cluster.y_position - (event_height / 2)))

    xmin = min(max(xmin, 0), image_width - event_width)
    ymin = min(max(ymin, 0), image_height - event_height)

    xmax = xmin + event_width
    ymax = ymin + event_height

    return ymin, ymax, xmin, xmax


def bounding_boxes_overlap(bounding_box_a, bounding_box_b):
    ymin_a, ymax_a, xmin_a, xmax_a = bounding_box_a
    ymin_b, ymax_b, xmin_b, xmax_b = bounding_box_b

    overlap_x = min(xmax_a, xmax_b) > max(xmin_a, xmin_b)
    overlap_y = min(ymax_a, ymax_b) > max(ymin_a, ymin_b)

    return overlap_x and overlap_y


def cut_event(image, ymin, ymax, xmin, xmax):

    return image[..., ymin:ymax, xmin:xmax]


def make_event_records(clusters):
    event_records = list()

    for index, cluster in enumerate(clusters, start=1):
        event_records.append(
            {
                "event_number": index,
                "cluster": cluster,
                "label": f"{index}: x={cluster.x_position:.1f}, y={cluster.y_position:.1f}",
            }
        )

    return event_records


def main(
    clusters_file,
    image_dir,
    num_events,
    event_height,
    event_width,
    odirname,
    display_channels,
    fiji_executable,
):

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    cluster_data = read_cluster_data(clusters_file)

    cluster_sample = select_subsample(cluster_data, num_events)

    img_paths = get_images_path(image_dir)
    img_paths_by_name = {path.stem: path for path in img_paths}
    event_records_by_image = dict()

    for img_name, clusters in cluster_sample.items():
        if img_name in img_paths_by_name:
            event_records_by_image[img_name] = make_event_records(clusters)

    image_cache = dict()
    saved_events_by_image = defaultdict(dict)

    def get_image(img_name):
        if img_name not in image_cache:
            image_cache[img_name] = read_image(img_paths_by_name[img_name])

        return image_cache[img_name]

    def get_event_context(img_name, event_index):
        image = get_image(img_name)
        event_record = event_records_by_image[img_name][event_index]
        cluster = event_record["cluster"]
        event_key = event_record["event_number"]

        ymin, ymax, xmin, xmax = get_cutting_position(
            image.shape[-2:],
            cluster,
            event_height,
            event_width,
        )
        bounding_box = (ymin, ymax, xmin, xmax)
        event_name = f"{img_name}_{event_key}_{bounding_box}"
        saved_class_name, saved_path = get_existing_saved_event(event_name)
        already_saved = saved_class_name is not None
        if saved_class_name:
            saved_events_by_image[img_name][event_key] = (bounding_box, saved_path)

        event = cut_event(image, ymin, ymax, xmin, xmax)
        overlaps_saved_box = any(
            previous_event_key != event_key
            and previous_saved_path.exists()
            and bounding_boxes_overlap(bounding_box, previous_bounding_box)
            for previous_event_key, (previous_bounding_box, previous_saved_path)
            in saved_events_by_image[img_name].items()
        )
        saved_message = ""
        overlap_message = ""

        if saved_class_name:
            saved_message = f"Already saved in {saved_class_name}: {saved_path.name}."

        if overlaps_saved_box:
            overlap_message = (
                "This event overlaps with a previously saved tile for this image."
            )

        return {
            "event": event,
            "event_name": event_name,
            "event_key": event_key,
            "bounding_box": bounding_box,
            "already_saved": already_saved,
            "saved_message": saved_message,
            "overlap_message": overlap_message,
        }

    def mark_saved(img_name, event_index, bounding_box, class_name):
        event_record = event_records_by_image[img_name][event_index]
        event_key = event_record["event_number"]
        event_name = f"{img_name}_{event_key}_{bounding_box}"
        saved_path = odirname / class_name / f"{event_name}.tif"
        saved_events_by_image[img_name][event_key] = (bounding_box, saved_path)

    def get_existing_saved_event(event_name):
        for class_name in ("bacteria", "background", "artifacts", "uncertain"):
            saved_path = odirname / class_name / f"{event_name}.tif"
            if saved_path.exists():
                return class_name, saved_path

        return None, None

    app = EventBrowserGui(
        event_records_by_image=event_records_by_image,
        get_event_context=get_event_context,
        mark_saved=mark_saved,
        main_folder=odirname,
        class_names=("bacteria", "background", "artifacts", "uncertain", "ignore"),
        display_channels=display_channels,
        fiji_executable=fiji_executable,
    )
    app.mainloop()

if __name__ == "__main__":
    main(
        CLUSTER_POSITIONS_FILE,
        IMAGES_DIR,
        NUMBER_EVENTS,
        EVENT_HEIGHT,
        EVENT_WIDTH,
        ODIRNAME,
        DISPLAY_CHANNELS,
        FIJI_EXECUTABLE,
    )
