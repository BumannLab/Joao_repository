from pathlib import Path
import itertools

import numpy as np
import pytest

from Scripts.compute_direction_and_cut_overlap import (
    compute_overlap_pixels,
    convert_row_numerical,
    cut_overlap,
    get_all_tiles_directions_to_remove_overlap,
    get_all_tiles_position_snake,
    obtain_direction_to_remove_overlap,
    obtain_tile_position,
)


class TestConvertRowNumerical:
    def test_valid_letters_len_1(self):
        valid_letters = "ABCDEFGHIJKLMNOPQRST"
        
        for char in valid_letters:
            assert convert_row_numerical(char) == ord(char) - ord("A")

    def test_non_valid_letters_len_1(self):
        non_valid_letters = "UVWXYZ"
        
        for char in non_valid_letters:
            with pytest.raises(AssertionError):
                assert convert_row_numerical(char)

    def test_valid_combinations(self):
        assert convert_row_numerical("AA") == 0
        assert convert_row_numerical("AT") == 19
        assert convert_row_numerical("BA") == 20
        assert convert_row_numerical("BT") == 39

    def test_non_valid_combinations(self):
        with pytest.raises(AssertionError):
            convert_row_numerical("AU")
            convert_row_numerical("AZ")
            convert_row_numerical("BY")
            convert_row_numerical("TA")


class TestObtainTilePosition:
    def test_with_string(self):
        assert obtain_tile_position("BA03") == (2, 20)

    def test_with_path(self):
        assert obtain_tile_position(Path("AA01.nd2")) == (0, 0)

class TestGetAllTilesPositionSnake:

    def test_normal_case_for_3x3_square(self):
        tile_names = [
            "01_AA01",
            "02_AA02",
            "03_AA03",
            "04_AB03",
            "05_AB02",
            "06_AB01",
            "07_AC01",
            "08_AC02",
            "09_AC03",
        ]
        
        result = get_all_tiles_position_snake(tile_names)

        assert result == [
            (0, 0), (1, 0), (2, 0),
            (2, 1), (1, 1), (0, 1),
            (0, 2), (1, 2), (2, 2),
        ]

    def test_duplicates_found(self):

        tile_names = [
            "01_AA01",
            "02_AA02",
            "03_AA03",
            "03_AA03",
            "04_AA03"
        ]

        with pytest.raises(AssertionError):
            get_all_tiles_position_snake(tile_names)


class TestObtainDirectionToRemoveOverlap:
    def test_top_left_tile(self):
        assert obtain_direction_to_remove_overlap((0, 0), 2) == [""]

    def test_even_row_non_first_column(self):
        assert obtain_direction_to_remove_overlap((2, 2), 3) == ["top", "left"]

    def test_odd_row_non_last_column(self):
        assert obtain_direction_to_remove_overlap((1, 1), 3) == ["top", "right"]

    def test_odd_row_last_column(self):
        assert obtain_direction_to_remove_overlap((3, 1), 3) == ["top"]

    def test_even_row_last_column(self):
        assert obtain_direction_to_remove_overlap((2, 2), 2) == ["top", "left"]


def test_get_all_tiles_directions_to_remove_overlap():
    tile_names = [
        "AA01",
        "AA02",
        "AB01",
        "AB02",
    ]

    result = get_all_tiles_directions_to_remove_overlap(tile_names)

    assert result == [[""], ["left"], ["top", "right"], ["top"]]



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
