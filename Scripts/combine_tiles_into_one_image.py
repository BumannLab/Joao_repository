import numpy as np
import tifffile as tiff

from utils import get_images_path, read_image, AXES

TILES_PER_IMAGE = 42

path =  "/data/joao/Materials/Joao_repository/Data/RAW_DATASET/B2022.10246/Threshold800_MeanFilter1000&250/uncertain" # "Joao_repository/Results/voxel_analysis/New_mTB_Data_4Channel_MAX/Images"
odirname = "/data/joao/Materials/Joao_repository"


img_path = get_images_path(path, ".nd2", ".tif", ".tiff")

for i in range(0,len(img_path), TILES_PER_IMAGE):
    image_name = f"{i}-{i+TILES_PER_IMAGE}_images.tif"

    combined_images = []
    count = 0
    for img in img_path[i:i+TILES_PER_IMAGE]:
        
        try:
            img = read_image(img)
        except:
            continue

        combined_images.append(img)
        count += 1
    
    combined_images = np.array(combined_images)

    axes = AXES[-combined_images.ndim:]

    tiff.imwrite(
        f"{odirname}/{image_name}",
        combined_images,
        imagej=True,
        metadata={"mode": "color", "axes": axes, "unit": "micron", "spacing":0.5},
        resolution=(5.884696082960086, 5.884696082960086)
    )

# Add multiprocessing so each processor combines tiles into a single ZCYX images
