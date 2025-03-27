# import os
# from PIL import Image
# import numpy as np

# # Define the color mapping
# COLOR_MAP = {
#     "G3_Mask": (255, 255, 0),  # Yellow
#     "G4_Mask": (255, 0, 0),    # Red
#     "Normal_Mask": (0, 255, 0), # Green
#     "Stroma_Mask": (255, 105, 180) # Pink
# }

# # Paths for input and output folders
# input_folder = "G:/sernetdata/AGGC2022_train/Subset1_Train_annotation/Train"
# output_folder = "G:/sernetdata/AGGC2022_train/Subset1_Train_annotation/Train2"

# # Ensure output folder exists
# os.makedirs(output_folder, exist_ok=True)

# def process_mask(mask_path, output_path, color):
#     # Open the image
#     image = Image.open(mask_path).convert("L")
    

#     # Convert to numpy array
#     mask_array = np.array(image)
    
#     # Create an RGB array with white background
#     rgb_array = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
#     rgb_array[mask_array == 0] = [255, 255, 255]  # White background
#     rgb_array[mask_array != 0] = color  # Apply mask color

#     # Convert back to PIL Image
#     colorized_image = Image.fromarray(rgb_array)

#     # Save the image
#     colorized_image.save(output_path)

# # Process each mask file
# for root, _, files in os.walk(input_folder):
#     for filename in files:
#         for mask_name, color in COLOR_MAP.items():
#             if mask_name in filename:
#                 input_path = os.path.join(root, filename)
#                 relative_path = os.path.relpath(root, input_folder)
#                 output_dir = os.path.join(output_folder, relative_path)
#                 os.makedirs(output_dir, exist_ok=True)
#                 output_path = os.path.join(output_dir, filename)
#                 process_mask(input_path, output_path, color)
#                 print(f"Processed: {filename} in {relative_path}")
#                 break





import os
from PIL import Image
import numpy as np

# Disable pixel limit check
Image.MAX_IMAGE_PIXELS = None

# Define the color mapping
COLOR_MAP = {
    "G3_Mask": (255, 255, 0),  # Yellow
    "G4_Mask": (255, 0, 0),    # Red
    "Normal_Mask": (0, 255, 0), # Green
    "Stroma_Mask": (255, 105, 180) # Pink
}

# Paths for input and output folders
input_folder = "G:/sernetdata/AGGC2022_train/Subset1_Train_annotation/TrainCopy"
output_folder = "D:/New folder"
#"G:\sernetdata\AGGC2022_train\Subset1_Train_annotation\TrainCopy"
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# def process_mask(mask_path, output_path, color):
#     # Open the image
#     image = Image.open(mask_path).convert("L")
    
#     # Convert to numpy array
#     mask_array = np.array(image)
    
#     # Create an RGB array with white background
#     rgb_array = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
#     rgb_array[mask_array == 0] = [255, 255, 255]  # White background
#     rgb_array[mask_array != 0] = color  # Apply mask color

#     # Convert back to PIL Image
#     colorized_image = Image.fromarray(rgb_array)

#     # Save the image
#     colorized_image.save(output_path)




def process_mask(mask_path, output_path, color):
    # Open the image
    image = Image.open(mask_path).convert("L")
    width, height = image.size

    # Create an output image with white background
    colorized_image = Image.new("RGB", (width, height), (255, 255, 255))  # White background

    # Process in chunks
    chunk_size = 2048  # Process 2048x2048 chunks
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            box = (x, y, min(x + chunk_size, width), min(y + chunk_size, height))
            chunk = image.crop(box)
            chunk_array = np.array(chunk)

            # Create a chunk RGB array
            rgb_chunk = np.zeros((*chunk_array.shape, 3), dtype=np.uint8)
            rgb_chunk[chunk_array == 0] = [255, 255, 255]  # White background
            rgb_chunk[chunk_array != 0] = color  # Apply mask color

            # Paste the processed chunk into the output image
            colorized_chunk = Image.fromarray(rgb_chunk)
            colorized_image.paste(colorized_chunk, box)

    # Save the final colorized image
    colorized_image.save(output_path)

# Process each mask file
for root, _, files in os.walk(input_folder):
    for filename in files:
        for mask_name, color in COLOR_MAP.items():
            if mask_name in filename:
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                process_mask(input_path, output_path, color)
                print(f"Processed: {filename} in {relative_path}")
                break
