

import os
from PIL import Image

def create_mask_patches(input_directory, output_directory, patch_size=(256, 256)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the output directory if it doesn't exist

    # Iterate over all mask images in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(input_directory, filename)
            try:
                with Image.open(file_path) as img:
                    # Ensure the image is of expected size
                    if img.size == (4608, 4608):
                        # Calculate number of patches in each dimension
                        num_patches_x = img.width // patch_size[0]
                        num_patches_y = img.height // patch_size[1]

                        # Generate patches
                        for i in range(num_patches_x):
                            for j in range(num_patches_y):
                                # Define the bounding box for each patch
                                left = i * patch_size[0]
                                upper = j * patch_size[1]
                                right = left + patch_size[0]
                                lower = upper + patch_size[1]

                                # Extract the patch
                                patch = img.crop((left, upper, right, lower))

                                # Construct filename for the patch
                                # Including original filename and coordinates to identify the patch
                                patch_filename = f"{filename.rstrip('.png')}_mask_patch_{i}_{j}.png"
                                patch_path = os.path.join(output_directory, patch_filename)

                                # Save the patch
                                # Save as PNG to avoid compression artifacts that might alter the mask
                                patch.save(patch_path, 'PNG')
                                print(f"Saved {patch_path}")
            except IOError:
                print(f"Error processing mask image {filename}")

# Specify the directory containing the mask images and the directory to save the patches
input_directory = "data2/Mixed_Masks2"
output_directory = "data2/Mixed_Masks_patches"

# Create mask patches
create_mask_patches(input_directory, output_directory)
