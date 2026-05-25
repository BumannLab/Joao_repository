from utils import create_luts

def extract_channel_colors_as_luts(nd2_metadata):
    channel_luts = list()

    channel_metadata = nd2_metadata.channels

    for channel in channel_metadata:
        rgba = channel.channel.color
        channel_luts.append(create_luts([rgba.r, rgba.g, rgba.b]))

    return channel_luts


def extract_voxel_size(nd2_metadata) -> dict:
    voxel_um = dict()

    # All channels have the same voxel size, so we can retrieve the information from any
    voxel_size_metadata = nd2_metadata.channels[0].volume.axesCalibration

    for axis, size in zip(["PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ"], voxel_size_metadata):
        voxel_um[axis] = float(size)

    return voxel_um


def get_resolution(voxel_size):

    psx = voxel_size["PhysicalSizeX"]
    psy = voxel_size["PhysicalSizeY"]
    resolution = (1.0 / psx, 1.0 / psy)

    return resolution