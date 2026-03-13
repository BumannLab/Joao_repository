import numpy as np
from glob import glob
import tifffile as tiff

TILES_PER_IMAGE = 100

path =  "Joao_repository/Results/voxel_analysis/Binary_Max_Images" # "Joao_repository/Results/voxel_analysis/New_mTB_Data_4Channel_MAX/Images"
img_path = sorted(glob(f"{path}/*.tif"))

for i in range(0,len(img_path), TILES_PER_IMAGE):
    image_name = f"Joao_repository/Results/voxel_analysis/New_mTB_Data_4Channel_MAX/combined_images/{i}-{i+TILES_PER_IMAGE}_images.tif"

    combined_images = []
    
    for img in img_path[i:i+TILES_PER_IMAGE]:
    
        img = tiff.imread(img)

        combined_images.append(img)

    tiff.imwrite(
        image_name,
        np.array(combined_images),
        imagej=True,
        metadata={"mode": "color", "axes": "ZCYX"},
    )

# Add multiprocessing so each processor combines tiles into a single ZCYX images
