import numpy as np
import pytest

from Pixel_Intensity_Thresholding_Analysis.pixel_intensity_thresholding import (find_pixel_clusters, 
    filter_clusters, 
    get_channel_intensity, 
    get_summed_intensities,
    get_all_pixel_positions)


class TestFindPixelClusters:
    def test_find_pixel_clusters_channel_1_intensity_greater_or_equal_than_10(self, image):
        threshold = 10
        channel = 0

        expected_results = {
            "Label": [1],
            "Area" : [19.0],
            "Intensity_max": [18.0]
        }

        result = find_pixel_clusters(image, channel, threshold)

        for i, prop in enumerate(result):
            assert prop.label == expected_results["Label"][i]
            assert prop.area == expected_results["Area"][i]
            assert prop.intensity_max == expected_results["Intensity_max"][i]

    def test_find_pixel_clusters_channel_4_intensity_greater_or_equal_than_15(self, image):
        threshold = 15
        channel = 3

        expected_results = {
            "Label": [1, 2, 3, 4],
            "Area" : [1.0, 4.0, 1.0, 1.0],
            "Intensity_max": [17.0, 19.0, 15.0, 17.0]
        }

        result = find_pixel_clusters(image, channel, threshold)

        for i, prop in enumerate(result):
            assert prop.label == expected_results["Label"][i]
            assert prop.area == expected_results["Area"][i]
            assert prop.intensity_max == expected_results["Intensity_max"][i]


class TestFilterClusters:
    def test_filter_clusters_cluster_size_greater_than_3(self, all_clusters):
        cluster_size_threshold = 3

        expected_results = {
            "Area" : [4.0],
            "Intensity_max": [19.0]
        }

        result = filter_clusters(all_clusters[0], cluster_size_threshold)

        assert len(result) == 1

        for i, prop in enumerate(result):
            assert prop.area == expected_results["Area"][i]
            assert prop.intensity_max == expected_results["Intensity_max"][i]

    def test_filter_clusters_cluster_size_greater_than_4(self, all_clusters):
        cluster_size_threshold = 4

        expected_results = []

        result = filter_clusters(all_clusters[0], cluster_size_threshold)

        assert len(result) == len(expected_results)


def test_get_channel_intensity(image, pixel_coords):

    expected_result = np.array([[3, 14, 0, 17],
                                [17, 12, 5, 0],
                                [9, 14, 17, 5],
                                [1, 6, 0, 11],
                                [4, 8, 19, 13]])

    result = get_channel_intensity(image, pixel_coords)

    assert (result==expected_result).all()


def test_get_summed_intensities(image, pixel_coords):
    expected_result = np.array(["34", "54", "41", "46"])

    result = get_summed_intensities(image, pixel_coords)
    
    assert (result==expected_result).all()


def test_get_all_pixel_positions(all_clusters):
    expected_result = np.array([[[0,0]],
                               [[0,2],[0,3],[1,2],[2,1]],
                               [[3,5]],
                                [[5,3]]])

    result = get_all_pixel_positions(all_clusters[1])

    assert (result==expected_result).all()
    
    
@pytest.fixture
def image():
    np.random.seed(5)
    return np.random.randint(0, 20, (4, 6, 6))

@pytest.fixture
def all_clusters(image):
    all_clusters = [find_pixel_clusters(image, 3, 15),
                    find_pixel_clusters(image, 0, 15)]
    return all_clusters

@pytest.fixture
def pixel_coords():
    return np.array([[0,0],
                     [2,2],
                     [4,1],
                     [4,3],
                     [5,5]])