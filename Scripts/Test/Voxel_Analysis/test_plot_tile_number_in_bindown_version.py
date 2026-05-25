import pytest

from plot_tile_number_in_bindown_version import adjusted_tile_center_pos


class TestAdjustedTileCenterPos:

    def test_adjusted_tile_center_pos_small_tiles_1x_bindown(self, all_tiles_positions):
        tile_height = 3
        tile_width = 3
        pixels_to_remove_h = 1
        pixels_to_remove_w = 1
        bindown_by = 1

        expected_result = [(1.5, 1.5), (3.5, 1.5), (5.5, 1.5),
                           (5.5, 3.5), (3.5, 3.5), (1.5, 3.5),
                           (1.5, 5.5), (3.5, 5.5), (5.5, 5.5)]
        
        result = adjusted_tile_center_pos(all_tiles_positions,
                                          tile_height,
                                          tile_width,
                                          pixels_to_remove_h,
                                          pixels_to_remove_w,
                                          bindown_by)
        
        assert result == expected_result
        

    def test_adjusted_tile_center_pos_40x_lens_5x_bindown(self, all_tiles_positions):
        tile_height = 2304
        tile_width = 2304
        pixels_to_remove_h = 115
        pixels_to_remove_w = 115
        bindown_by = 5

        expected_result = [(230.4, 230.4), (668.2, 230.4), (1106, 230.4),
                           (1106, 668.2), (668.2, 668.2), (230.4, 668.2),
                           (230.4, 1106), (668.2, 1106), (1106, 1106)]
        
        result = adjusted_tile_center_pos(all_tiles_positions,
                                          tile_height,
                                          tile_width,
                                          pixels_to_remove_h,
                                          pixels_to_remove_w,
                                          bindown_by)
        
        assert result == expected_result
    


@pytest.fixture()
def all_tiles_positions():

    number_rows, number_cols = 3,3

    tiles_positions = list()
    forward = True

    for i in range(number_rows):

        if forward:

            for j in range(number_cols):
                tiles_positions.append((j, i))
        
        else:
            for j in range(number_cols-1, -1, -1):
                tiles_positions.append((j, i))
                
        if forward:
            forward = False
        else:
            forward = True

    return tiles_positions