import cv2
import os
import numpy as np

# Set the path to the directory containing the images
source_folder = "G:/New_VOC_Label_Patches"
destination_folder = "G:/Filtered_Green_Images"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Define the range for green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Process each file in the source directory
for file_name in os.listdir(source_folder):
    if file_name.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        # Read the image
        path = os.path.join(source_folder, file_name)
        image = cv2.imread(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Check if there is a significant amount of green
        if cv2.countNonZero(mask) > (0.1 * mask.size):  # Adjust the 0.1 threshold as needed
            # Copy the image to the destination folder
            destination_path = os.path.join(destination_folder, file_name)
            cv2.imwrite(destination_path, image)
            print(f"Copied {file_name}")

print("Completed filtering and copying green images.")
############################################################################################################################################################################



import cv2
import os
import numpy as np

# Paths to the directories
mask_folder = 'G:/New_VOC_Label_Patches'
train_image_folder = 'C:/Users/Nishith/Downloads/New_train_Patches'
destination_folder =  "G:/Filtered_Green_Train_Images"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Define the range for green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Process each file in the mask directory
for file_name in os.listdir(mask_folder):
    if file_name.endswith(".png"):  # Check for PNG mask files
        base_name = os.path.splitext(file_name)[0]  # Remove the file extension
        corresponding_train_image_name = base_name + ".jpg"  # Append the JPG extension for the training image

        # Read the mask image
        mask_path = os.path.join(mask_folder, file_name)
        mask_image = cv2.imread(mask_path)
        hsv = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)

        # Create a mask for green color
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Check if there is a significant amount of green
        if cv2.countNonZero(green_mask) > (0.1 * green_mask.size):  # Adjust the 0.1 threshold as needed
            # Find the corresponding training image
            corresponding_train_image_path = os.path.join(train_image_folder, corresponding_train_image_name)
            if os.path.exists(corresponding_train_image_path):
                # Copy the corresponding training image to the destination folder
                destination_path = os.path.join(destination_folder, corresponding_train_image_name)
                cv2.imwrite(destination_path, cv2.imread(corresponding_train_image_path))
                print(f"Copied {corresponding_train_image_name}")

print("Completed filtering and copying training images with green masks.")





import os

# Paths to the directories
# mask_folder = 'path/to/mask/folder'
# train_image_folder = 'path/to/train/image/folder'
mask_folder = "G:/Filtered_Green_Images"
train_image_folder = "G:/Filtered_Green_Train_Images"
# Get the list of mask files without extension
mask_files = {os.path.splitext(file)[0] for file in os.listdir(mask_folder) if file.endswith(".png")}
# Get the list of training image files without extension
train_image_files = {os.path.splitext(file)[0] for file in os.listdir(train_image_folder) if file.endswith(".jpg")}

# Determine extra and missing files
extra_masks = mask_files - train_image_files
missing_train_images = train_image_files - mask_files

# Print the results
if extra_masks:
    print("Extra mask files:", extra_masks)
else:
    print("No extra mask files.")

if missing_train_images:
    print("Missing training images:", missing_train_images)
else:
    print("No missing training images.")










import cv2
import os
import numpy as np

# Paths to the directories
mask_folder = "G:/Filtered_Green_Images"
train_image_folder = "G:/Filtered_Green_Train_Images"

# mask_folder = 'path/to/mask/folder'
# train_image_folder = 'path/to/train/image/folder'
destination_folder = "G:/Filtered_Green_Images1"
destination_image_folder = "G:/Filtered_Green_Train_Images1"

# Create the destination folders if they don't exist
os.makedirs(destination_folder, exist_ok=True)
os.makedirs(destination_image_folder, exist_ok=True)

# Define the range for green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Process each file in the mask directory
for file_name in os.listdir(mask_folder):
    if file_name.endswith(".png"):  # Check for PNG mask files
        base_name = os.path.splitext(file_name)[0]  # Remove the file extension
        corresponding_train_image_name = base_name + ".jpg"  # Append the JPG extension for the training image

        # Read the mask image
        mask_path = os.path.join(mask_folder, file_name)
        mask_image = cv2.imread(mask_path)
        hsv = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)

        # Create a mask for green color
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply the mask to get only the green parts
        result = cv2.bitwise_and(mask_image, mask_image, mask=green_mask)

        # Save the green-only mask
        cleaned_mask_path = os.path.join(destination_folder, file_name)
        cv2.imwrite(cleaned_mask_path, result)

        # Copy the corresponding training image if the green mask is significant
        if cv2.countNonZero(green_mask) > (0.1 * green_mask.size):  # Adjust the 0.1 threshold as needed
            # Check and copy the corresponding training image
            corresponding_train_image_path = os.path.join(train_image_folder, corresponding_train_image_name)
            destination_train_image_path = os.path.join(destination_image_folder, corresponding_train_image_name)
            if os.path.exists(corresponding_train_image_path):
                cv2.imwrite(destination_train_image_path, cv2.imread(corresponding_train_image_path))

print("Completed processing for green-only masks and corresponding training images.")





import os

# Paths to the directories
mask_folder = 'G:/Filtered_Green_Images1'
train_image_folder = 'G:/Filtered_Green_Train_Images1'

# Get the base names of the mask files (without extensions)
mask_files = {os.path.splitext(file)[0] for file in os.listdir(mask_folder) if file.endswith(".png")}

# Process each file in the training image directory
for file_name in os.listdir(train_image_folder):
    if file_name.endswith((".jpg", ".jpeg")):  # Check for JPEG training images
        base_name = os.path.splitext(file_name)[0]  # Remove the file extension

        # Check if there is no corresponding mask
        if base_name not in mask_files:
            # Build the path to the training image file
            train_image_path = os.path.join(train_image_folder, file_name)
            # Remove the training image as it has no corresponding mask
            os.remove(train_image_path)
            print(f"Removed {file_name} as it has no corresponding mask.")

print("Cleanup completed. Removed training images without corresponding masks.")




import os

def clean_directory(folder_path, keep_count=1000):
    # List all files in the directory and sort them
    files = sorted(os.listdir(folder_path))
    
    # Split into files to keep and files to delete
    files_to_keep = files[:keep_count]
    files_to_delete = files[keep_count:]
    
    # Delete the files beyond the first 1000
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        print(f"Deleted {file_name} from {folder_path}")

# Paths to the directories
train_image_folder = 'G:/Filtered_Green_Train_Images1 - Copy'
mask_image_folder = 'G:/Filtered_Green_Images1 - Copy'

# Apply the cleaning function to both directories
clean_directory(train_image_folder)
clean_directory(mask_image_folder)

print("Directory cleanup completed.")



G:/Filtered_Green_Images1 - Copy
G:/Filtered_Green_Train_Images1 - Copy


















import cv2
import os
import numpy as np

def convert_to_bw_mask(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Process each file in the source directory
    for file_name in os.listdir(source_folder):
        if file_name.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
            # Read the image
            path = os.path.join(source_folder, file_name)
            image = cv2.imread(path)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for green color
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Convert mask to a binary image
            bw_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # This converts it to a BGR image if needed elsewhere
            bw_mask = cv2.cvtColor(bw_mask, cv2.COLOR_BGR2GRAY)  # Convert to single channel grayscale image

            # Save the black-and-white mask
            destination_path = os.path.join(destination_folder, file_name)
            cv2.imwrite(destination_path, bw_mask)
            print(f"Converted and saved black-and-white mask for {file_name}")

# Paths to the directories
source_folder = 'G:/Filtered_Green_Images1 - Copy'
destination_folder = 'G:/Filtered_Green_Images1 - Copy11'

# Convert masks and save them to the destination folder
convert_to_bw_mask(source_folder, destination_folder)

print("Conversion to black-and-white masks completed.")




import cv2
import os
import numpy as np

def convert_to_bw_mask(source_folder, destination_folder, num_images_to_print=5):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Counter to track the number of images for which to print details
    print_count = 0

    # Process each file in the source directory
    for file_name in os.listdir(source_folder):
        if file_name.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
            # Read the image
            path = os.path.join(source_folder, file_name)
            image = cv2.imread(path)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for green color
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Convert mask to a binary image
            bw_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # This converts it to a BGR image if needed elsewhere
            bw_mask = cv2.cvtColor(bw_mask, cv2.COLOR_BGR2GRAY)  # Convert to single channel grayscale image

            # Save the black-and-white mask
            destination_path = os.path.join(destination_folder, file_name)
            cv2.imwrite(destination_path, bw_mask)
            print(f"Converted and saved black-and-white mask for {file_name}")

            # Print the shape and channel information for a limited number of images
            if print_count < num_images_to_print:
                print(f"Shape and channel info for {file_name}: {image.shape}")
                print_count += 1

# Paths to the directories
source_folder = 'G:/Filtered_Green_Images1 - Copy'
destination_folder = 'G:/Filtered_Green_Images1 - Copy11'

# Convert masks and save them to the destination folder, printing shape for the first few images
convert_to_bw_mask(source_folder, destination_folder)

print("Conversion to black-and-white masks and printing of image details completed.")








############################################################################################################################################################################
import os
import shutil

# Define the source and destination directories
source_dir = '/home/pratibha/VIT/data2/Train_Imgs'  # Path to the folder with JPG files
mask_dir = '/home/pratibha/VIT/data2/VOC_level_green'    # Path to the folder with PNG masks
destination_dir = '/home/pratibha/VIT/data2/New_train_Patches_filtered'  # Path where matching JPG files will be copied

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get a list of mask filenames in the PNG directory and strip the file extension
mask_files = {os.path.splitext(file)[0] for file in os.listdir(mask_dir) if file.endswith('.png')}

# Loop through all files in the source directory
for file in os.listdir(source_dir):
    # Check if the file is a JPG and its name (without extension) is in the mask_files set
    if file.endswith('.jpg') and os.path.splitext(file)[0] in mask_files:
        # Construct full file paths
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        
        # Copy file
        shutil.copy(source_file, destination_file)
        print(f'Copied {file} to {destination_dir}')

print("Copying complete!")
########### 
from PIL import Image
import os

def generate_patches(img_path, mask_path, patch_size, img_save_path, mask_save_path):
    # Load the image and mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    # Calculate the number of patches in each dimension
    width, height = img.size
    x_patches = width // patch_size
    y_patches = height // patch_size

    # Create directories if they don't exist
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    # Generate patches
    for i in range(x_patches):
        for j in range(y_patches):
            # Define the box to crop (left, upper, right, lower)
            box = (i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size)

            # Crop the image and mask
            img_patch = img.crop(box)
            mask_patch = mask.crop(box)

            # Save the patches
            img_patch.save(os.path.join(img_save_path, f'img_patch_{i}_{j}.png'))
            mask_patch.save(os.path.join(mask_save_path, f'mask_patch_{i}_{j}.png'))

# Parameters
img_dir = 'data2/New_train_Patches_Filtered_green'  # Path to the original images
mask_dir = 'data2/VOC_level_green'  # Path to the corresponding masks
img_patches_dir = 'data2/New_train_Patches_Filtered_green_patch'  # Path to save image patches
mask_patches_dir = 'data2/VOC_level_green_patch'  # Path to save mask patches
patch_size = 256  # Size of each patch

# Process all images and masks
for img_filename in os.listdir(img_dir):
    if img_filename.endswith('.jpg'):
        mask_filename = img_filename.replace('.jpg', '.png')  # Assuming mask has same name but .png extension
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            generate_patches(img_path, mask_path, patch_size, img_patches_dir, mask_patches_dir)
            print(f'Processed {img_filename} and {mask_filename}')

print("Patch generation complete!")




import os

def rename_files(directory, old_substring, new_substring):
    # Iterate through all the files in the directory
    for filename in os.listdir(directory):
        if old_substring in filename:
            # Construct the new filename by replacing the old substring with the new one
            new_filename = filename.replace(old_substring, new_substring)
            # Create full file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed {filename} to {new_filename}')

# Specify the directory containing the images
directory = 'data2/VOC_level_green_patch'
# Specify the substrings to replace
old_substring = 'mask'
new_substring = 'img'

# Call the function to rename the files
rename_files(directory, old_substring, new_substring)
print("File renaming complete!")




from PIL import Image
import os

def convert_masks_to_black_and_white(directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all the files in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the masks are in PNG format
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open the image
            img = Image.open(file_path)

            # Convert image to grayscale ('L' mode)
            img = img.convert('L')

            # Process the image to be either black or white
            # 255 (white) if the pixel value is 1, 0 (black) otherwise
            threshold = 1  # since class value 1 should be white, others black
            img = img.point(lambda p: 255 if p == threshold else 0, '1')

            # Save the converted image
            img.save(output_path)
            print(f'Processed {filename}')

# Directory paths
input_directory = 'data2/VOC_level_green_patch'
output_directory = 'data2/VOC_level_green_patch_classvalue'

# Convert masks
convert_masks_to_black_and_white(input_directory, output_directory)
print("Conversion complete!")



from PIL import Image
import os

def convert_green_masks(directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all the files in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the masks are in PNG format
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open the image
            img = Image.open(file_path).convert('RGBA')

            # Create a new image to store the black and white version
            bw_img = Image.new('1', img.size)  # '1' for 1-bit pixels, black and white

            # Process the image to set green pixels to white and others to black
            pixels = img.load()
            bw_pixels = bw_img.load()
            for i in range(img.width):
                for j in range(img.height):
                    r, g, b, a = pixels[i, j]
                    if g == 255 and r == 0 and b == 0:  # Assuming pure green is class 1
                        bw_pixels[i, j] = 255  # Set to white
                    else:
                        bw_pixels[i, j] = 0  # Set to black

            # Save the converted image
            bw_img.save(output_path)
            print(f'Processed {filename}')

# Directory paths
input_directory = 'data2/VOC_level_green_patch'
output_directory = 'data2/VOC_level_green_patch_classvalue'

# Convert masks
convert_green_masks(input_directory, output_directory)
print("Conversion complete!")




from PIL import Image
import os

def is_green(pixel):
    r, g, b = pixel[:3]  # Ignore alpha if present
    # Check if green is the dominant color and significantly higher than red and blue
    return g > r + 50 and g > b + 50 and g > 128

def convert_green_masks(directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all the files in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the masks are in PNG format
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open the image
            img = Image.open(file_path)
            img = img.convert('RGB')  # Convert to RGB to simplify pixel access

            # Create a new image to store the black and white version
            bw_img = Image.new('1', img.size)  # '1' for 1-bit pixels, black and white

            # Process the image to set green pixels to white and others to black
            pixels = img.load()
            bw_pixels = bw_img.load()
            for i in range(img.width):
                for j in range(img.height):
                    if is_green(pixels[i, j]):
                        bw_pixels[i, j] = 255  # Set to white
                    else:
                        bw_pixels[i, j] = 0  # Set to black

            # Save the converted image
            bw_img.save(output_path)
            print(f'Processed {filename}')

# Directory paths
input_directory = 'data2/VOC_level_green_patch'
output_directory = 'data2/VOC_level_green_patch_classvalue1'

# Convert masks
convert_green_masks(input_directory, output_directory)
print("Conversion complete!")





from PIL import Image
import os

def analyze_image(file_path):
    # Load the image
    img = Image.open(file_path)
    img = img.convert('RGB')  # Ensure it's in RGB

    # Initialize counters
    total_pixels = 0
    green_pixels = 0

    # Process the image to count green pixels
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]
            if g > r and g > b:  # Simplistic check for green dominance
                green_pixels += 1
            total_pixels += 1

    # Calculate the percentage of green pixels
    green_percentage = (green_pixels / total_pixels) * 100

    # Return results
    return {
        "total_pixels": total_pixels,
        "green_pixels": green_pixels,
        "green_percentage": green_percentage
    }

# File path to the uploaded mask image
file_path = 'data2/VOC_level_green_patch/img_patch_16_6.png'

# Analyze the image
results = analyze_image(file_path)
print("Analysis Results:")
print(f"Total Pixels: {results['total_pixels']}")
print(f"Green Pixels: {results['green_pixels']}")
print(f"Percentage of Green Pixels: {results['green_percentage']:.2f}%")





from PIL import Image
import os
import colorsys

def is_green(pixel):
    # Convert RGB to HSV
    r, g, b = pixel[:3]
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    # Define green in HSV: broader range for hue, and reasonable ranges for saturation and value
    return (0.25 <= h <= 0.45) and s >= 0.3 and v >= 0.2

def convert_green_masks(directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all the files in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the masks are in PNG format
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open the image
            img = Image.open(file_path).convert('RGB')  # Ensure it's in RGB

            # Create a new image to store the black and white version
            bw_img = Image.new('1', img.size)  # '1' for 1-bit pixels, black and white

            # Process the image to set green pixels to white and others to black
            pixels = img.load()
            bw_pixels = bw_img.load()
            for i in range(img.width):
                for j in range(img.height):
                    if is_green(pixels[i, j]):
                        bw_pixels[i, j] = 255  # Set to white
                    else:
                        bw_pixels[i, j] = 0  # Set to black

            # Save the converted image
            bw_img.save(output_path)
            print(f'Processed {filename}')

# Directory paths
input_directory = 'data2/VOC_level_green_patch'
output_directory = 'data2/VOC_level_green_patch_classvalue11'

# Convert masks
convert_green_masks(input_directory, output_directory)
print("Conversion complete!")




import os

def remove_substring_from_filenames(folder, substring, file_ext='.png'):
    # Traverse through the folder
    for filename in os.listdir(folder):
        if filename.endswith(file_ext):
            # Check if the substring is in the filename
            if substring in filename:
                # Construct the full old file path
                old_file_path = os.path.join(folder, filename)
                # Replace the substring in the filename
                new_filename = filename.replace(substring, '')
                # Construct the full new file path
                new_file_path = os.path.join(folder, new_filename)
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} to {new_file_path}")

# Example usage
folder = 'data2/Train_images2_patches'  # Path to the folder containing the files
substring = '.j'  # Substring to remove from the filenames

remove_substring_from_filenames(folder, substring)
