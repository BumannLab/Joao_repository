from pathlib import Path

import numpy as np
import pytest

from compute_direction_and_cut_overlap import (
    compute_overlap_pixels,
    convert_row_numerical,
    cut_overlap,
    get_all_tiles_directions_to_remove_overlap,
    get_all_tiles_position_snake,
    obtain_direction_to_remove_overlap,
    obtain_tile_position,
)


class TestConvertRowNumerical:
    def test_single_letter_rows(self):
        assert convert_row_numerical("A") == 0
        assert convert_row_numerical("B") == 1
        assert convert_row_numerical("Z") == 25

    def test_multiple_letter_rows(self):
        assert convert_row_numerical("AA") == 0
        assert convert_row_numerical("BA") == 26
        assert convert_row_numerical("AB") == 1


class TestObtainTilePosition:
    def test_with_string(self):
        assert obtain_tile_position("BA03") == (2, 26)

    def test_with_path(self):
        assert obtain_tile_position(Path("AA01.nd2")) == (0, 0)


def test_get_all_tiles_position_snake():
    tile_names = [
        "AA01",
        "AA02",
        "AA03",
        "AB01",
        "AB02",
        "AB03",
        "AC01",
        "AC02",
        "AC03",
    ]

    result = get_all_tiles_position_snake(tile_names)

    assert result == [
        (0, 0), (1, 0), (2, 0),
        (2, 1), (1, 1), (0, 1),
        (0, 2), (1, 2), (2, 2),
    ]


def test_get_all_tiles_position_snake_preserves_duplicates():
    tile_names = [
        "AF26_1",
        "AF26_2",
        "AF27_1",
        "AF27_2",
        "AG26_1",
        "AG26_2",
    ]

    result = get_all_tiles_position_snake(tile_names)

    assert result == [
        (25, 5), (25, 5), (26, 5), (26, 5),
        (25, 6), (25, 6),
    ]


class TestObtainDirectionToRemoveOverlap:
    def test_top_left_tile(self):
        assert obtain_direction_to_remove_overlap((0, 0), 2) == [""]

    def test_even_row_non_first_column(self):
        assert obtain_direction_to_remove_overlap((2, 2), 3) == ["top", "left"]

    def test_odd_row_non_last_column(self):
        assert obtain_direction_to_remove_overlap((1, 1), 3) == ["top", "right"]

    def test_odd_row_last_column(self):
        assert obtain_direction_to_remove_overlap((3, 1), 3) == ["top"]


def test_get_all_tiles_directions_to_remove_overlap():
    tile_names = [
        "AA01",
        "AA02",
        "AB01",
        "AB02",
    ]

    result = get_all_tiles_directions_to_remove_overlap(tile_names)

    assert result == [[""], ["left"], ["top", "right"], ["top"]]


def test_get_all_tiles_directions_to_remove_overlap_preserves_duplicates():
    tile_names = [
        "AF26_1",
        "AF26_2",
        "AF27_1",
        "AF27_2",
        "AG26_1",
        "AG26_2",
    ]

    result = get_all_tiles_directions_to_remove_overlap(tile_names)

    assert result == [
        [""],
        [""],
        ["left"],
        ["left"],
        ["top"],
        ["top"],
    ]


class TestComputeOverlapPixels:
    def test_nearest_rounding(self):
        assert compute_overlap_pixels(5, 2304, 2304) == (115, 115)

    def test_floor_rounding(self):
        assert compute_overlap_pixels(12.5, 10, 10, rounding_mode="floor") == (1, 1)

    def test_ceil_rounding(self):
        assert compute_overlap_pixels(12.5, 10, 10, rounding_mode="ceil") == (2, 2)


class TestCutOverlap:
    def test_cuts_top(self):
        image = np.arange(16).reshape(4, 4)

        result = cut_overlap(image, 1, 0, ["top"])

        np.testing.assert_array_equal(result, image[1:, :])

    def test_cuts_left(self):
        image = np.arange(16).reshape(4, 4)

        result = cut_overlap(image, 0, 1, ["left"])

        np.testing.assert_array_equal(result, image[:, 1:])

    def test_cuts_right(self):
        image = np.arange(16).reshape(4, 4)

        result = cut_overlap(image, 0, 1, ["right"])

        np.testing.assert_array_equal(result, image[:, :-1])

    def test_cuts_top_and_left_on_3d_image(self):
        image = np.arange(2 * 4 * 4).reshape(2, 4, 4)

        result = cut_overlap(image, 1, 1, ["top", "left"])

        np.testing.assert_array_equal(result, image[:, 1:, 1:])


# Bug report
# `obtain_tile_position` does not validate that the tile name matches the expected
# pattern before calling `match.groups()`, so malformed names will raise an
# `AttributeError` instead of a clearer exception.
