import numpy as np
import pytest

from ZOrganized_scripts.NeuralNetworkSegmentationConfig.transforms import build_transform_pipeline

class TestBuildTransformPipeline


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