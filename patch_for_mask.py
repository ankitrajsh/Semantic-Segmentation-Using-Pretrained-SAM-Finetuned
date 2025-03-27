import os
import numpy as np
from PIL import Image
import tifffile as tiff
from patchify import patchify

# Disable decompression bomb warning in Pillow
Image.MAX_IMAGE_PIXELS = None

# Folder containing mask files
mask_folder = r'G:\sernetdata\AGGC2022_train\Subset1_Train_annotation\Train\Subset1_Train_3'

# Output directory for mask patches
os.makedirs('patches/masks2', exist_ok=True)

# Mask classes to be processed
mask_classes = ['G3_Mask', 'G4_Mask', 'Normal_Mask', 'Stroma_Mask']


# Handle masks for each mask class
for mask_class in mask_classes:
    mask_file = os.path.join(mask_folder, f'{mask_class}.tif')
    
    # Check if the mask file exists
    if os.path.exists(mask_file):
        try:
            # Try reading the mask image using Pillow first
            with Image.open(mask_file) as img:
                # Convert to numpy array
                large_mask = np.array(img)

                # Check the dimensions of the mask
                mask_height, mask_width = large_mask.shape
                if mask_height < 256 or mask_width < 256:
                    print(f"Mask {mask_class} dimensions are smaller than patch size (256x256). Padding will be applied.")

                    # Pad the mask to ensure it is at least 256x256
                    pad_height = max(0, 256 - mask_height)
                    pad_width = max(0, 256 - mask_width)

                    # Apply padding to make the mask size compatible
                    large_mask = np.pad(large_mask, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
                    print(f"Padded mask {mask_class} to size: {large_mask.shape}")

                # Create a folder for each mask class
                mask_output_folder = f'patches/masks/{mask_class}'
                os.makedirs(mask_output_folder, exist_ok=True)

                # Patchify the mask into 256x256 patches
                patches_mask = patchify(large_mask, (256, 256), step=256)

                # Save the mask patches to the corresponding mask class folder
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        patch_filename = os.path.join(mask_output_folder, f'{mask_class}_{i}{j}.tif')
                        tiff.imwrite(patch_filename, single_patch_mask)
                        # Normalize the mask to [0, 1] (if necessary for further processing)
                        single_patch_mask = single_patch_mask / 255.0

        except Exception as e:
            print(f"Error reading {mask_class}: {e}")
    else:
        print(f"Mask file {mask_file} does not exist.")
