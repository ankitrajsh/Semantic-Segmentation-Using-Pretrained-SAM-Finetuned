
!pip install datasets
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q monai
!pip install matplotlib

from datasets import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import SamModel, SamProcessor
from monai.losses import DiceCELoss
from tqdm import tqdm
from statistics import mean

#patches mixed 
# Assuming you have paths to your image and mask directories
image_dir = "data2/Train_images2_patches"
mask_dir = "data2/Mixed_Masks_patches"

# Retrieve the list of image and mask filenames
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Sort the lists to ensure corresponding images and masks align (if filenames are matching)
image_files.sort()
mask_files.sort()

# Function to load images safely
def load_images(files, directory):
    images = []
    for file in files:
        with Image.open(os.path.join(directory, file)) as img:
            images.append(img.copy())  # Important to use copy() to keep the image after closing the file
    return images

# Load images and masks into a dictionary
dataset_dict = {
    "image": load_images(image_files, image_dir),
    "label": load_images(mask_files, mask_dir)
}
from datasets import Dataset

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

dataset


# # Choose a random index from the dataset
# img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
# example_image = dataset[img_num]["image"]
# example_mask = dataset[img_num]["label"]

# # Plotting the images
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Plot the first image on the left
# axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
# axes[0].set_title("Image")

# # Plot the second image (mask) on the right
# axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
# axes[1].set_title("Mask")

# # Hide axis ticks and labels
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])

# # Display the images side by side
# plt.show()
########    
import matplotlib.pyplot as plt
import numpy as np
import random

# Assuming dataset is already loaded and contains 'image' and 'label' keys for images and masks

# Choose a random index from the dataset
img_num = random.randint(0, len(dataset) - 1)  # Adjust based on your dataset's structure
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the image on the left
axes[0].imshow(example_image, cmap='gray')  # No need to convert if already in the correct format
axes[0].set_title("Image")

# Plot the mask on the right
axes[1].imshow(example_mask, cmap='gray')  # No need to convert if already in the correct format
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.axis('off')  # This hides the axes ticks and labels more efficiently

# Display the images side by side
plt.show()




# class SAMDataset(Dataset):
#     """
#     This class is used to create a dataset that serves input images and masks.
#     It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#     """
#     # Initialize the dataset and processor
#     def __init__(self, dataset, processor):
#         self.dataset = dataset
#         self.processor = processor
#     # Override the __len__ method to return the length of the dataset
#     def __len__(self):
#         return len(self.dataset)
#     #
#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = item["image"]
#         ground_truth_mask = np.array(item["label"])
#         prompt = self.get_bounding_box(ground_truth_mask)

#         # Even if bounding box is default (empty), still process and return structured data
#         inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
#         inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#         inputs["ground_truth_mask"] = ground_truth_mask
#         return inputs

#     def get_bounding_box(self, ground_truth_map):
#         y_indices, x_indices = np.where(ground_truth_map > 0)
#         if y_indices.size == 0 or x_indices.size == 0:
#             # Return a default bounding box that indicates an empty or invalid box
#             return [0, 0, 1, 1]  # Small, non-zero area at the origin
      

#         # Compute bounding box coordinates
#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)

#         # Add random perturbation to bounding box coordinates
#         H, W = ground_truth_map.shape
#         x_min = max(0, x_min - np.random.randint(0, 20))
#         x_max = min(W, x_max + np.random.randint(0, 20))
#         y_min = max(0, y_min - np.random.randint(0, 20))
#         y_max = min(H, y_max + np.random.randint(0, 20))

#         return [x_min, y_min, x_max, y_max]
#temp+delete
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor, scale_masks=False):
        """
        Initializes the dataset, processor, and mask scaling option.
        Args:
        - dataset: A dataset object which provides the images and masks.
        - processor: A processing function or object to apply to each dataset item.
        - scale_masks: A boolean to determine if mask values should be scaled from 1 to 255.
        """
        self.dataset = dataset
        self.processor = processor
        self.scale_masks = scale_masks

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.
        Args:
        - idx: Index of the data item to retrieve.
        Returns:
        - A dictionary containing processed image, mask, and bounding box information.
        """
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # Scale mask to 255 if required (useful for certain visualizations and processing)
        if self.scale_masks:
            display_mask = ground_truth_mask * 255
        else:
            display_mask = ground_truth_mask

        prompt = self.get_bounding_box(ground_truth_mask)

        # Process the image and bounding box for model input
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(display_mask, dtype=torch.long)

        return inputs

    # def get_bounding_box(self, ground_truth_map):
    #     """
    #     Calculates a bounding box from the mask where pixel values are greater than zero.
    #     Args:
    #     - ground_truth_map: A 2D numpy array representing the mask.
    #     Returns:
    #     - A list representing the bounding box coordinates [x_min, y_min, x_max, y_max].
    #     """
    #     y_indices, x_indices = np.where(ground_truth_map > 0)
    #     if y_indices.size == 0 or x_indices.size == 0:
    #         # Return a default bounding box that indicates an empty or invalid box
    #         return [0, 0, 1, 1]  # Small, non-zero area at the origin

    #     # Compute bounding box coordinates
    #     x_min, x_max = np.min(x_indices), np.max(x_indices)
    #     y_min, y_max = np.min(y_indices), np.max(y_indices)

    #     # Add random perturbation to bounding box coordinates
    #     H, W = ground_truth_map.shape
    #     x_min = max(0, x_min - np.random.randint(0, 20))
    #     x_max = min(W, x_max + np.random.randint(0, 20))
    #     y_min = max(0, y_min - np.random.randint(0, 20))
    #     y_max = min(H, y_max + np.random.randint(0, 20))

    #     return [x_min, y_min, x_max, y_max]
    def get_bounding_box(self, ground_truth_map):
        """
        Calculates a bounding box from the mask where pixel values are greater than zero.
        Assumes ground_truth_map is a 2D numpy array representing the mask.
        Returns a list representing the bounding box coordinates [x_min, y_min, x_max, y_max].
        """
        # Ensure the mask is 2D
        if ground_truth_map.ndim > 2:
            # Assuming the class labels are in the last channel
            ground_truth_map = ground_truth_map[..., 0]

        y_indices, x_indices = np.where(ground_truth_map > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            return [0, 0, 1, 1]  # Small, non-zero area at the origin

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]

#############
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset

# class SAMDataset(Dataset):
#     """
#     This class is used to create a dataset that serves input images and masks with specific class-color encoding.
#     It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#     """
#     def __init__(self, dataset, processor, visualize_colors=False):
#         """
#         Initializes the dataset, processor, and color visualization option.
#         Args:
#         - dataset: A dataset object which provides the images and masks.
#         - processor: A processing function or object to apply to each dataset item.
#         - visualize_colors: A boolean to determine if masks should be visualized with specific colors.
#         """
#         self.dataset = dataset
#         self.processor = processor
#         self.visualize_colors = visualize_colors

#     def __len__(self):
#         """Returns the length of the dataset."""
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         """
#         Retrieves an item from the dataset by index.
#         Args:
#         - idx: Index of the data item to retrieve.
#         Returns:
#         - A dictionary containing processed image, mask, and bounding box information.
#         """
#         item = self.dataset[idx]
#         image = item["image"]
#         ground_truth_mask = np.array(item["label"])

#         if self.visualize_colors:
#             display_mask = self.apply_color_map(ground_truth_mask)
#         else:
#             display_mask = ground_truth_mask

#         prompt = self.get_bounding_box(ground_truth_mask)

#         # Process the image and bounding box for model input
#         inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
#         inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#         inputs["ground_truth_mask"] = torch.tensor(display_mask, dtype=torch.uint8 if self.visualize_colors else torch.long)

#         return inputs

#     def apply_color_map(self, mask):
#         """
#         Applies a color map to the mask where different classes are represented by different colors.
#         Args:
#         - mask: A 2D numpy array representing the mask.
#         Returns:
#         - A 3D numpy array (RGB) representing the mask with specific colors for each class.
#         """
#         color_map = {
#             0: [0, 0, 0],      # Black for background
#             1: [255, 0, 0],    # Red for class 1
#             2: [0, 255, 0],    # Green for class 2
#             3: [255, 255, 0]   # Yellow for class 3
#         }
#         # Create an RGB image with the same size as the mask
#         rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

#         for class_value, color in color_map.items():
#             rgb_mask[mask == class_value] = color

#         return rgb_mask

#     def get_bounding_box(self, ground_truth_map):
#         """
#         Calculates a bounding box from the mask where pixel values are greater than zero.
#         Args:
#         - ground_truth_map: A 2D numpy array representing the mask.
#         Returns:
#         - A list representing the bounding box coordinates [x_min, y_min, x_max, y_max].
#         """
#         y_indices, x_indices = np.where(ground_truth_map > 0)
#         if y_indices.size == 0 or x_indices.size == 0:
#             return [0, 0, 1, 1]  # Small, non-zero area at the origin

#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)

#         H, W = ground_truth_map.shape
#         x_min = max(0, x_min - np.random.randint(0, 20))
#         x_max = min(W, x_max + np.random.randint(0, 20))
#         y_min = max(0, y_min - np.random.randint(0, 20))
#         y_max = min(H, y_max + np.random.randint(0, 20))

#         return [x_min, y_min, x_max, y_max]
##########

# from torch.utils.data import Dataset
# import numpy as np
# import torch

# class SAMDataset(Dataset):
#     """
#     This class is used to create a dataset that serves input images and masks for objects marked as green, red, or yellow.
#     It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
#     """
#     def __init__(self, dataset, processor, scale_masks=False):
#         """
#         Initializes the dataset, processor, and mask scaling option.
#         Args:
#         - dataset: A dataset object which provides the images and masks.
#         - processor: A processing function or object to apply to each dataset item.
#         - scale_masks: A boolean to determine if mask values should be scaled from 1 to 255.
#         """
#         self.dataset = dataset
#         self.processor = processor
#         self.scale_masks = scale_masks

#     def __len__(self):
#         """Returns the length of the dataset."""
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         """
#         Retrieves an item from the dataset by index.
#         Args:
#         - idx: Index of the data item to retrieve.
#         Returns:
#         - A dictionary containing processed image, mask, and bounding box information.
#         """
#         item = self.dataset[idx]
#         image = item["image"]
#         ground_truth_mask = np.array(item["label"])

#         # Scale mask to 255 if required (useful for certain visualizations and processing)
#         display_mask = ground_truth_mask * (255 if self.scale_masks else 1)

#         # Process the image for model input
#         inputs = self.processor(image, return_tensors="pt")
#         inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#         inputs["ground_truth_mask"] = torch.tensor(display_mask, dtype=torch.long)

#         # Retrieve bounding boxes for green, red, and yellow objects
#         green_bbox = self.get_bounding_box(ground_truth_mask, 1)
#         red_bbox = self.get_bounding_box(ground_truth_mask, 2)
#         yellow_bbox = self.get_bounding_box(ground_truth_mask, 3)

#         inputs["green_bbox"] = green_bbox
#         inputs["red_bbox"] = red_bbox
#         inputs["yellow_bbox"] = yellow_bbox

#         return inputs

#     def get_bounding_box(self, ground_truth_map, class_value):
#         """
#         Calculates a bounding box for specified class from the mask.
#         Args:
#         - ground_truth_map: A 2D numpy array representing the mask.
#         - class_value: The integer value in the mask corresponding to the class of interest.
#         Returns:
#         - A list representing the bounding box coordinates [x_min, y_min, x_max, y_max].
#         """
#         y_indices, x_indices = np.where(ground_truth_map == class_value)
#         if y_indices.size == 0 or x_indices.size == 0:
#             # Return a default bounding box that indicates an empty or invalid box
#             return [0, 0, 1, 1]  # Small, non-zero area at the origin

#         # Compute bounding box coordinates
#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)

#         # Add random perturbation to bounding box coordinates (optional)
#         H, W = ground_truth_map.shape
#         x_min = max(0, x_min - np.random.randint(0, 20))
#         x_max = min(W, x_max + np.random.randint(0, 20))
#         y_min = max(0, y_min - np.random.randint(0, 20))
#         y_max = min(H, y_max + np.random.randint(0, 20))

#         return [x_min, y_min, x_max, y_max]

#temp+delete

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]

# Check if the example is None, which might be the case if the bounding box was empty
if example is None:
    print("The first item in the dataset is None (possibly an empty bounding box).")
else:
    # If the example is not None, proceed to print shapes of each component
    for k, v in example.items():
        print(f"{k}: {v.shape}")



# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

# Iterating through the DataLoader to get batches
for batch in train_dataloader:
    print(batch["ground_truth_mask"].shape)
    break
    # Continue with processing or any other operations


batch["ground_truth_mask"].shape


# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

from torch.optim import Adam
# import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
 

def dice_coefficient(pred, target):
    smooth = 1.0  # Smooth factor to avoid division by zero
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
def iou_score(pred, target, smooth=1e-6):
    # Convert predictions to binary (0 or 1) using a threshold (typically 0.5 for binary masks)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # Flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Intersection and Union
    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection
    
    # IoU
    IoU = (intersection + smooth) / (union + smooth)
    return IoU
import torch
from tqdm import tqdm
from statistics import mean

# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Transfer model to the chosen device
model.to(device)
model.train()

num_epochs = 4
for epoch in range(num_epochs):
    epoch_losses = []  # to store loss values
    epoch_dice = []  # to store dice scores
    epoch_iou = []  # to store IoU scores
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch in progress_bar:
        # Forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # Compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = DiceCELoss()(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Compute Dice Coefficient and IoU
        dice_score = dice_coefficient(torch.sigmoid(predicted_masks), ground_truth_masks)
        iou_score_val = iou_score(predicted_masks, ground_truth_masks)
        epoch_dice.append(dice_score.item())
        epoch_iou.append(iou_score_val.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for this batch and update tqdm description
        epoch_losses.append(loss.item())
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}, Batch Dice: {dice_score:.4f}, Batch IoU: {iou_score_val:.4f}")

    # After all batches, print average loss, dice, and IoU for the epoch
    average_loss = mean(epoch_losses)
    average_dice = mean(epoch_dice)
    average_iou = mean(epoch_iou)
    print(f'Epoch {epoch+1}/{num_epochs}, Mean Loss: {average_loss:.4f}, Mean Dice: {average_dice:.4f}, Mean IoU: {average_iou:.4f}')
torch.save(model.state_dict(), "G:\sernetdata\model_100_datasets_4.pth")


print(ground_truth_map.shape)  # This will tell you the number of dimensions and the size of each dimension.






#visulize the result
import torch
from transformers import SamModel
import numpy as np
# Path to your model file
model_path = "G:\sernetdata\model_100_datasets_4.pth"

# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)




from PIL import Image
from transformers import SamProcessor

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Encapsulate in another list
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs


def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

def load_mask(mask_path):
    """Load a mask from the specified path."""
    mask = Image.open(mask_path)
    return np.array(mask)

def segment_image(image_path):
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()



 # Move the mask to CPU and convert to NumPy array
image_path = "C:/Users/Nishith/Downloads/New_train_Patches/slide001_core004_patch_12_1.jpg"
predicted_mask = segment_image(image_path)


# Visualizing the result
import matplotlib.pyplot as plt
# Correctly preparing the mask for visualization
plt.imshow(predicted_mask[0].squeeze(), cmap='gray')  # Use .squeeze() to remove the singular dimension
plt.title("Segmented Mask")
plt.axis('off')
plt.show()



#test the model
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your model file
model_path = "G:/sernetdata/model_100_datasets_4.pth"

# Load the model and set it to evaluation mode
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()

# Define device and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    """Prepare input image for the model."""
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Full image as bounding box
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs

def segment_image(image_path):
    """Segment an image and return the predicted mask."""
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()

# Specify the image path
image_path = "G:/Filtered_Green_Train_Images1/slide003_core080_patch_5_4.jpg"
predicted_mask = segment_image(image_path)

# Visualizing the segmented mask
plt.imshow(predicted_mask[0].squeeze(), cmap='gray')  # Use .squeeze() to remove the singular dimension
plt.title("Segmented Mask")
plt.axis('off')
plt.show()






import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your model file
model_path = "G:/sernetdata/model_100_datasets_4.pth"

# Load the model and set it to evaluation mode
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()

# Define device and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    """Prepare input image for the model."""
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Full image as bounding box
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs

def segment_image(image_path):
    """Segment an image and return the predicted mask."""
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()

# Specify the image path
image_path = "G:/Filtered_Green_Train_Images1/slide003_core080_patch_5_4.jpg"
predicted_mask = segment_image(image_path)

# Convert to binary mask
binary_mask = predicted_mask[0] > 0.5  # Apply threshold

# Visualizing the binary mask
plt.imshow(binary_mask, cmap='gray')  # Show the mask as black and white
plt.title("Segmented Mask")
plt.axis('off')
plt.show()



import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your model file
model_path = "G:/sernetdata/model_100_datasets_4.pth"

# Load the model and set it to evaluation mode
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()

# Define device and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    """Prepare input image for the model."""
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Full image as bounding box
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs

def segment_image(image_path):
    """Segment an image and return the predicted mask."""
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()

# Specify the image path
image_path = "G:/Filtered_Green_Train_Images/slide001_core003_patch_3_14.jpg"
predicted_mask = segment_image(image_path)
# Convert to binary mask
binary_mask = (predicted_mask[0] > 0.5).astype(int)  # Apply threshold and convert to int for clean binary mask

# Visualizing the binary mask
# Ensure you use `.squeeze()` to remove any singleton dimensions, this time explicitly making sure it's 2D
plt.imshow(binary_mask.squeeze(), cmap='gray')  # Show the mask as black and white
plt.title("Segmented Mask")
plt.axis('off')
plt.show()





################
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your model file
model_path = "G:/sernetdata/model_100_datasets_4.pth"

# Load the model and set it to evaluation mode
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()

# Define device and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    """Prepare input image for the model."""
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Full image as bounding box
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs

def segment_image(image_path):
    """Segment an image and return the predicted mask."""
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

# Paths to the image and its corresponding mask
image_path = "G:/Filtered_Green_Train_Images/slide001_core007_patch_2_12.jpg"
mask_path = "G:/Filtered_Green_Images/slide001_core007_patch_2_12.png"  # Adjust this path as necessary

# Load images
original_image = load_image(image_path)
original_mask = load_image(mask_path)

# Get predicted mask
predicted_mask = segment_image(image_path)
binary_mask = (predicted_mask[0] > 0.5).astype(int)  # Convert to binary mask

# Visualize all three images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(np.array(original_mask), cmap='gray')
axes[1].set_title("Original Mask")
axes[1].axis('off')

axes[2].imshow(binary_mask, cmap='gray')
axes[2].set_title("Predicted Mask")
axes[2].axis('off')

plt.show()


import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your model file
model_path = "G:/sernetdata/model_100_datasets_4.pth"

# Load the model and set it to evaluation mode
model = SamModel.from_pretrained("facebook/sam-vit-base", state_dict=torch.load(model_path))
model.eval()

# Define device and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def prepare_input(image_path):
    """Prepare input image for the model."""
    image = Image.open(image_path).convert("RGB")
    input_boxes = [[[0, 0, image.width, image.height]]]  # Full image as bounding box
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    return inputs

def segment_image(image_path):
    """Segment an image and return the predicted mask."""
    inputs = prepare_input(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    predicted_masks = torch.sigmoid(outputs.pred_masks).squeeze(1)  # Apply sigmoid and remove batch dimension

    return predicted_masks.cpu().numpy()

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

# Paths to the image and its corresponding mask
image_path = "G:/Filtered_Green_Train_Images/slide001_core007_patch_2_3.jpg"
mask_path = "G:/Filtered_Green_Images/slide001_core007_patch_2_3.png"  # Adjust this path as necessary

# Load images
original_image = load_image(image_path)
original_mask = load_image(mask_path)

# Get predicted mask
predicted_mask = segment_image(image_path)
binary_mask = (predicted_mask[0] > 0.7).astype(int)  # Convert to binary mask

# Ensure the mask is 2D
binary_mask = binary_mask.squeeze()

# Visualize all three images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(np.array(original_mask), cmap='gray')
axes[1].set_title("Original Mask")
axes[1].axis('off')

axes[2].imshow(binary_mask, cmap='gray')  # Ensure binary_mask is 2D
axes[2].set_title("Predicted Mask")
axes[2].axis('off')

plt.show()

