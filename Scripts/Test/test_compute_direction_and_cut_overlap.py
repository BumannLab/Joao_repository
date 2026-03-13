import pytest

from utils import get_images_path
from compute_direction_and_cut_overlap import get_all_tiles_position_snake

def test_get_all_tiles_position_corrected():

    data_paths = get_images_path("/data/joao/mTBData/20251127/full111/1", ".nd2")

    expected_result = []

    for y in range(19):
        if y % 2 == 0:
            for x in range(60):
                expected_result.append((x, y))
        else:
            for x in range(59, -1, -1):
                expected_result.append((x, y))

    result = get_all_tiles_position_snake(data_paths)

    assert result == expected_result

