
from datasets import Dataset
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt  # Ensure this import is included
import random
from tqdm import tqdm
from statistics import mean
import torch 
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
from torch.optim import Adam
import monai

!pip install datasets
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q monai
!pip install matplotlib


# Assuming you have paths to your image and mask directories
image_dir = 'C:/Users/Nishith/Downloads/New_train_Patches'
mask_dir = 'G:/New_VOC_Label_Patches'
# Path to the directory containing mask files

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

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)
dataset




# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()


class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = self.get_bounding_box(ground_truth_mask)

        # Even if bounding box is default (empty), still process and return structured data
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs


    def get_bounding_box(self, ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            # Return a default bounding box that indicates an empty or invalid box
            return [0, 0, 1, 1]  # Small, non-zero area at the origin
      

        # Compute bounding box coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add random perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]


# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create the SAMDataset instance
traindataset = SAMDataset(dataset=dataset, processor=processor)


# Check the shape of the first example in the dataset
example = train_dataset[0]

# Check if the example is None, which might be the case if the bounding box was empty
if example is None:
    print("The first item in the dataset is None (possibly an empty bounding box).")
else:
    # If the example is not None, proceed to print shapes of each component
    for k, v in example.items():
        print(f"{k}: {v.shape}")



# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=False)

# Assuming 'train_dataset' is already defined and is an instance of a Dataset class

# Iterating through the DataLoader to get batches
for batch in train_dataloader:
    print(batch["ground_truth_mask"].shape)
    break
    # Continue with processing or any other operations


batch["ground_truth_mask"].shape


# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
 
# training  start 

# Assuming 'model', 'train_dataloader', 'seg_loss', 'optimizer' are defined elsewhere
# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Transfer model to the chosen device
model.to(device)
model.train()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    epoch_losses = []  # to store loss values
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch in progress_bar:
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for this batch and update tqdm description
        epoch_losses.append(loss.item())
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}")

    # After all batches, print average loss for the epoch
    average_loss = mean(epoch_losses)
    print(f'Epoch {epoch+1}/{num_epochs}, Mean Loss: {average_loss:.4f}')

# training parts end
#save the model to load later
torch.save(model.state_dict(), "G:\sernetdata\model_checkpoint_10_EH.pth")

##############################################################################################################


# let's take a random training example
idx = random.randint(0, filtered_images.shape[0]-1)

# load image
test_image = dataset[idx]["image"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

my_mito_model.eval()

# forward pass
with torch.no_grad():
    outputs = my_mito_model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the first image on the left
axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(medsam_seg, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Plot the second image on the right
axes[2].imshow(medsam_seg_prob)  # Assuming the second image is grayscale
axes[2].set_title("Probability Map")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()





# Test part here 

import torch
from transformers import SamModel
import numpy as np
# Path to your model file
model_path = "G:/sernetdata/model_checkpoint.pth"

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
image_path = "G:/sernetdata/AGGC2022_train/Processed5/Images/patch_121_104_Subset1_Train_1.tiff"
predicted_mask = segment_image(image_path)

# Visualizing the result
import matplotlib.pyplot as plt
# Correctly preparing the mask for visualization
plt.imshow(predicted_mask[0].squeeze(), cmap='gray')  # Use .squeeze() to remove the singular dimension
plt.title("Segmented Mask")
plt.axis('off')
plt.show()



def plot_images_and_masks(image_paths, mask_paths):
    num_images = len(image_paths)
    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.array([axes])  # Convert to a 2D array for consistent indexing

    for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load and display the original image
        image = load_image(image_path)
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # Load and display the ground truth mask
        mask = load_mask(mask_path)
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth Mask')
        axes[idx, 1].axis('off')

        # Predict and display the mask
        predicted_mask = segment_image(image_path)
        axes[idx, 2].imshow(predicted_mask[0].squeeze(), cmap='gray')
        axes[idx, 2].set_title('Predicted Mask')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()




image_paths = [
    "G:/sernetdata/AGGC2022_train/Processed5/Images/patch_101_106_Subset1_Train_1.tiff"
]
mask_paths = [
    "G:/sernetdata/AGGC2022_train/Processed5/Masks/patch_101_106_Subset1_Train_1_mask.tiff"
]

# Call the function with your paths
plot_images_and_masks(image_paths, mask_paths)