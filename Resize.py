from PIL import Image
import os

# Set the path to the source and destination folders
source_folder = "C:/Users/Nishith/Downloads/New_train_Patches_Filtered"

destination_folder = "C:/Users/Nishith/Downloads/New_train_Patches_Filtered_RSIZED"


# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Resize each image in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image_path = os.path.join(source_folder, filename)
        img = Image.open(image_path)
        img = img.resize((4608, 4608), Image.ANTIALIAS)  # Resize the image
        
        # Save the resized image to the destination folder
        img.save(os.path.join(destination_folder, filename))

print("All images have been resized and saved.")
###
from PIL import Image
import os

# Set the path to the source and destination folders
source_folder = "C:/Users/Nishith/Downloads/VOC_level_green"

destination_folder = "C:/Users/Nishith/Downloads/VOC_level_green_RSIZED"
# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Resize each image in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image_path = os.path.join(source_folder, filename)
        img = Image.open(image_path)
        img = img.resize((4608, 4608), Image.Resampling.LANCZOS)  # Resize the image using the LANCZOS resampling algorithm
        
        # Save the resized image to the destination folder
        img.save(os.path.join(destination_folder, filename))

print("All images have been resized and saved.")