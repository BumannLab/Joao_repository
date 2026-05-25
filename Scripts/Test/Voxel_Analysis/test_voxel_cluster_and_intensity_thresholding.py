import pytest
import numpy as np

from Voxel_Analysis.voxel_cluster_and_intensity_thresholding import (build_out_file_sufix,
                                                                    voxel_channel_intensity_threshold,
                                                                    multi_channel_threshold_mask,
                                                                    )


class TestBuildOutFileSufix:
    def test_build_out_file_sufix_with_correct_condition(self, threshold_conditions):

        expected_result = ("SYBRGold_greater_200_DAPI_less_equal_100"
                           "_CD68AL555_greater_equal_100_LAMAL647_less_600")

        result = build_out_file_sufix(threshold_conditions[0])

        assert result == expected_result

    def test_build_out_file_sufix_with_incorrect_condition(self, threshold_conditions):

        expected_result = ("SYBRGold_greater_200_DAPI?100"
                           "_CD68AL555=100_LAMAL647&600")
        
        result = build_out_file_sufix(threshold_conditions[1])

        assert result == expected_result


class TestVoxelChannelIntensityThreshold:
    def test_voxel_channel_intensity_threshold_greater_operator(self, image):
        channel = 3
        operator = ">"
        threshold = 8

        expected_result = np.array([[[False, True, False],
                                    [True, True, False],
                                    [False, True, False]],
                                    
                                    [[False, True, True],
                                     [False, False, True],
                                     [True, True, False]]])
        
        result = voxel_channel_intensity_threshold(image, channel, operator, threshold)

        assert (result == expected_result).all()

    def test_voxel_channel_intensity_threshold_less_equal_operator(self, image):
        channel = 3
        operator = "<="
        threshold = 5

        expected_result = np.array([[[True, False, False],
                                    [False, False, True],
                                    [True, False, True]],
                                    
                                    [[False, False, False],
                                     [True, True, False],
                                     [False, False, False]]])
        
        result = voxel_channel_intensity_threshold(image, channel, operator, threshold)

        assert (result == expected_result).all()


class TestMultiChannelThresholdMask:
    def test_multi_channel_threshold_mask_two_thresholds(self, image):
        thresholds = [(0, "<=", 10),
                      (2, ">", 10)]
        
        expected_result = np.array([[[True, False, False],
                                    [True, False, False],
                                    [False, False, False]],
                                    
                                    [[False, False, False],
                                     [False, False, False],
                                     [True, True, False]]])

        result = multi_channel_threshold_mask(image, thresholds)

        assert (result == expected_result).all()

    def test_multi_channel_threshold_mask_one_threshold(self, image):
        thresholds = [(1, "<", 13)]
        
        expected_result = np.array([[[False, False, True],
                                    [True, False, False],
                                    [True, False, True]],
                                    
                                    [[False, False, False],
                                     [True, True, True],
                                     [False, True, True]]])

        result = multi_channel_threshold_mask(image, thresholds)

        assert (result == expected_result).all()
    

@pytest.fixture
def threshold_conditions():
    threshold_conditions_lst = [
        "SYBRGold>200,DAPI<=100,CD68AL555>=100,LAMAL647<600",
        "SYBRGold>200,DAPI?100,CD68AL555=100,LAMAL647&600"
    ]
    return threshold_conditions_lst

@pytest.fixture
def image():
    np.random.seed(5)
    return np.random.randint(0, 20, (2, 4, 3, 3))