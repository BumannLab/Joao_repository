import numpy as np
import pytest

import ZOrganized_scripts.NeuralNetworkSegmentationConfig.data as data_module


class TestReadInput:
    def test_2d_image_gets_z_singleton(self, mock_read_image, image_2d):
        mock_read_image(image_2d)

        result = data_module.read_input("image_2d.tif")

        assert result.shape == (1, 2, 2, 3)
        assert result.dtype == np.uint16

    def test_3d_image_keeps_shape(self, mock_read_image, image_3d):
        mock_read_image(image_3d)

        result = data_module.read_input("image_3d.tif")

        assert result.shape == (2, 2, 2, 3)
        assert result.dtype == np.uint16


class TestReadMask:
    def test_2d_mask_gets_z_and_c_singleton(self, mock_read_image, mask_2d):
        mock_read_image(mask_2d)

        result = data_module.read_mask("mask_2d.tif")

        assert result.shape == (1, 1, 2, 3)
        assert result.dtype == np.uint8

    def test_3d_mask_gets_c_singleton(self, mock_read_image, mask_3d):
        mock_read_image(mask_3d)

        result = data_module.read_mask("mask_3d.tif")

        assert result.shape == (2, 1, 2, 3)
        assert result.dtype == np.uint8


@pytest.fixture
def mock_read_image(monkeypatch):
    def apply_mock(image):
        def fake_read_image(path):
            return image

        monkeypatch.setattr(data_module, "read_image", fake_read_image)

    return apply_mock


@pytest.fixture
def image_2d():
    return np.array(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            [
                [6, 7, 8],
                [9, 10, 11],
            ],
        ],
        dtype=np.uint16,
    )

@pytest.fixture
def mask_2d():
    return np.array(
        [
                [0, 0, 0],
                [0, 1, 1],
        ],
        dtype=np.uint8,
    )

@pytest.fixture
def image_3d():
    return np.array(
        [
            [
                [
                    [0, 1, 2],
                    [3, 4, 5],
                ],
                [
                    [6, 7, 8],
                    [9, 10, 11],
                ],
            ],
            [
                [
                    [12, 13, 14],
                    [15, 16, 17],
                ],
                [
                    [18, 19, 20],
                    [21, 22, 23],
                ],
            ],
        ],
        dtype=np.uint16,
    )

@pytest.fixture
def mask_3d():
    return np.array(
        [
            [
                    [0, 0, 0],
                    [0, 1, 1],
            ],

           [
                    [0, 1, 1],
                    [0, 1, 1],
            ],
        ],
        dtype=np.uint8,
    )

# import numpy as np
# import pytest

# from ZOrganized_scripts.NeuralNetworkSegmentationConfig.data import read_input, read_mask


# class TestReadInput:
#     def test_2d_image_gets_z_singleton(self, image_2d):

#         result = read_input("ZOrganized_scripts/NeuralNetworkSegmentationConfig/Tests/data/input_2d.tif")

#         assert result.shape == (1, 4, 2, 3)
#         assert result.dtype == np.uint16
#         np.testing.assert_array_equal(result, image_2d)

#     def test_3d_image_keeps_shape(self, image_3d):

#         result = read_input("ZOrganized_scripts/NeuralNetworkSegmentationConfig/Tests/data/input_3d.tif")

#         assert result.shape == (5, 4, 2, 3)
#         assert result.dtype == np.uint16
#         np.testing.assert_array_equal(result, image_3d)



# class TestReadMask:
#     def test_2d_mask_gets_z_and_c_singleton(self, mask_2d):

#         result = read_mask("ZOrganized_scripts/NeuralNetworkSegmentationConfig/Tests/data/mask_2d.tif")

#         assert result.shape == (1, 1, 2, 3)
#         assert result.dtype == np.uint8
#         np.testing.assert_array_equal(result, mask_2d)


#     def test_3d_mask_gets_c_singleton(self, mask_3d):
    
#         result = read_mask("ZOrganized_scripts/NeuralNetworkSegmentationConfig/Tests/data/mask_3d.tif")
#         assert result.shape == (5, 1, 2, 3)
#         assert result.dtype == np.uint8
#         np.testing.assert_array_equal(result, mask_3d)





# @pytest.fixture
# def image_2d():
#     return np.arange(1, 4 * 2 * 3 + 1, dtype=np.uint16).reshape(1, 4, 2, 3)

# @pytest.fixture
# def image_3d():
#     return np.arange(1, 5 * 4 * 2 * 3 + 1, dtype=np.uint16).reshape(5, 4, 2, 3)

# @pytest.fixture
# def mask_2d():
#     return np.array([[[[0, 255, 0],
#                          [255, 0, 255]]]], dtype=np.uint8)

# @pytest.fixture
# def mask_3d():
#     return np.array([
#     [[[0, 255, 0],
#       [255, 0, 255]]],

#     [[[255, 255, 0],
#       [0, 255, 0]]],

#     [[[0, 0, 255],
#       [255, 255, 0]]],

#     [[[255, 0, 0],
#       [0, 255, 255]]],

#     [[[0, 255, 255],
#       [255, 0, 0]]]
# ], dtype=np.uint8)

