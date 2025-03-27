# setup.py

# Install required packages
!pip install datasets
!pip install matplotlib
!pip install -q monai
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install -q git+https://github.com/huggingface/transformers.git


from datasets import Dataset
from PIL import Image
import os

# Function to load images safely
def load_images(files, directory):
    images = []
    for file in files:
        with Image.open(os.path.join(directory, file)) as img:
            images.append(img.copy())  # Important to use copy() to keep the image after closing the file
    return images

# Load Image, sort and store it into dictionary
image_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Images'
mask_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Masks'

# Retrieve the list of image and mask filenames
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Sort the lists to ensure corresponding images and masks align
image_files.sort()
mask_files.sort()

# Load images and masks into a dictionary
dataset_dict = {
    "image": load_images(image_files, image_dir),
    "label": load_images(mask_files, mask_dir)
}
# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

import torch
print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


!nvcc --version