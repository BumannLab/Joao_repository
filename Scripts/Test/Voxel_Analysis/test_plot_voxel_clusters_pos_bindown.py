import pytest

from plot_clusters_pos_bindown import Cluster

# Assume there are 30 columns and 15 rows. 0-based (0 - 29, 0 - 14)

class TestCluster:
    def test_bindowned_rescaled_centroid_pos_example_1(self, cluster, tile_params):
        cluster = cluster[0]
        tile_column = 14
        tile_row = 7
        
        expected_result = Cluster(6234, 3439)

        cluster.bindowned_rescaled_centroid_pos(tile_column, tile_row, **tile_params)

        # Make sure it does not exceed beyond the possible values of tile column
        assert ((round((tile_params["tile_width"] - tile_params["pixels_remove_width"] ) / tile_params["bindown_by"]) * 29) + 
                round(tile_params["tile_width"] /  tile_params["bindown_by"])) >= expected_result.X_centroid

        # Make sure it does not exceed beyond the possible values of tile row 
        assert ((round((tile_params["tile_height"] - tile_params["pixels_remove_height"] ) / tile_params["bindown_by"]) * 14) + 
                round(tile_params["tile_height"] /  tile_params["bindown_by"])) >= expected_result.Y_centroid

        assert cluster == expected_result


    def test_bindowned_rescaled_centroid_pos_example_2_bottom_left(self, cluster, tile_params):
        cluster = cluster[1]
        tile_column = 29
        tile_row = 14
        
        expected_result = Cluster(13163, 6593)

        cluster.bindowned_rescaled_centroid_pos(tile_column, tile_row, **tile_params)

        # Make sure it does not exceed beyond the possible values of tile column
        assert ((round((tile_params["tile_width"] - tile_params["pixels_remove_width"] ) / tile_params["bindown_by"]) * 29) + 
                round(tile_params["tile_width"] /  tile_params["bindown_by"])) >= expected_result.X_centroid

        # Make sure it does not exceed beyond the possible values of tile row 
        assert ((round((tile_params["tile_height"] - tile_params["pixels_remove_height"] ) / tile_params["bindown_by"]) * 14) + 
                round(tile_params["tile_height"] /  tile_params["bindown_by"])) >= expected_result.Y_centroid

        assert cluster == expected_result

    
    def test_bindowned_rescaled_centroid_pos_example_3_within_overlap_region(self, cluster, tile_params):
        cluster = cluster[2]
        tile_column = 8
        tile_row = 2
        
        expected_result = Cluster(3504, 876)

        cluster.bindowned_rescaled_centroid_pos(tile_column, tile_row, **tile_params)

        # Make sure it does not exceed beyond the possible values of tile column
        assert ((round((tile_params["tile_width"] - tile_params["pixels_remove_width"] ) / tile_params["bindown_by"]) * 29) + 
                round(tile_params["tile_width"] /  tile_params["bindown_by"])) >= expected_result.X_centroid

        # Make sure it does not exceed beyond the possible values of tile row 
        assert ((round((tile_params["tile_height"] - tile_params["pixels_remove_height"] ) / tile_params["bindown_by"]) * 14) + 
                round(tile_params["tile_height"] /  tile_params["bindown_by"])) >= expected_result.Y_centroid

        assert cluster == expected_result


@pytest.fixture
def cluster():
    return [Cluster(511, 1864),
            Cluster(2304, 2304),
            Cluster(1, 1)]


@pytest.fixture
def tile_params():
        return {"tile_height" : 2304,
        "tile_width" : 2304,
        "pixels_remove_width" : 115,
        "pixels_remove_height" : 115,
        "bindown_by": 5}