import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import glob
import time
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        image = self.transform(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1  # Convert (1,2,3) → (0,1,2) for training

        return image, mask

# Load Image and Mask Paths
image_dir = "C:/Users/Nishith/Downloads/New_train_Patches"
mask_dir = "G:/New_VOC_Label_Patches"
image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))

# Split into Training and Validation Sets
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Create Dataloaders
train_dataset = SegmentationDataset(train_images, train_masks)
val_dataset = SegmentationDataset(val_images, val_masks)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# Load pre-trained model
num_classes = 4  # Background (0) + Classes (1,2,3)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512", 
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# Freeze all layers except the classification head
for param in model.segformer.parameters():
    param.requires_grad = False

# Only set requires_grad=True for parameters in the classification head
for param in model.decode_head.parameters():
    param.requires_grad = True

model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

def calculate_accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def print_metrics(epoch, num_epochs, batch, num_batches, loss, accuracy, phase):
    print(f"Epoch [{epoch+1}/{num_epochs}], {phase.capitalize()} Batch [{batch+1}/{num_batches}], "
          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# num_epochs = 10
# best_val_loss = float('inf')

# for epoch in range(num_epochs):
#     print(f"\nEpoch {epoch+1}/{num_epochs}")
#     print("-" * 20)

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#         else:
#             model.eval()

#         running_loss = 0.0
#         running_accuracy = 0.0
#         epoch_start_time = time.time()

#         dataloader = train_loader if phase == 'train' else val_loader

#         for batch, (images, masks) in enumerate(tqdm(dataloader, desc=phase.capitalize())):
#             images, masks = images.to(device), masks.to(device)

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(images).logits
#                 loss = criterion(outputs, masks)

#                 if phase == 'train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#             preds = torch.argmax(outputs, dim=1)
#             acc = calculate_accuracy(preds, masks)

#             running_loss += loss.item() * images.size(0)
#             running_accuracy += acc * images.size(0)

#             if batch % 10 == 0:  # Print every 10 batches
#                 print_metrics(epoch, num_epochs, batch, len(dataloader), loss.item(), acc, phase)

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_accuracy = running_accuracy / len(dataloader.dataset)
#         epoch_time = time.time() - epoch_start_time

#         print(f"{phase.capitalize()} Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
#         print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, {phase.capitalize()} Accuracy: {epoch_accuracy:.4f}")

#         if phase == 'val' and epoch_loss < best_val_loss:
#             best_val_loss = epoch_loss
#             torch.save(model.state_dict(), "best_model.pth")
#             print("Saved new best model!")

#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Current learning rate: {current_lr}")

# print("Training completed!")
# ############################################################################################################

# import time
# from tqdm import tqdm

# num_epochs = 10
# best_val_loss = float('inf')

# for epoch in range(num_epochs):
#     print(f"\nEpoch {epoch+1}/{num_epochs}")
#     print("-" * 20)

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#         else:
#             model.eval()

#         running_loss = 0.0
#         running_accuracy = 0.0
#         epoch_start_time = time.time()

#         dataloader = train_loader if phase == 'train' else val_loader

#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase.capitalize()} Epoch {epoch+1}")
#         for batch, (images, masks) in pbar:
#             images, masks = images.to(device), masks.to(device)

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(images).logits
#                 loss = criterion(outputs, masks)

#                 if phase == 'train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#             preds = torch.argmax(outputs, dim=1)
#             acc = calculate_accuracy(preds, masks)

#             running_loss += loss.item() * images.size(0)
#             running_accuracy += acc * images.size(0)

#             # Update progress bar
#             avg_loss = running_loss / ((batch + 1) * images.size(0))
#             avg_acc = running_accuracy / ((batch + 1) * images.size(0))
#             pbar.set_postfix({
#                 'loss': f'{avg_loss:.4f}',
#                 'accuracy': f'{avg_acc:.4f}'
#             })

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_accuracy = running_accuracy / len(dataloader.dataset)
#         epoch_time = time.time() - epoch_start_time

#         print(f"{phase.capitalize()} Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
#         print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, {phase.capitalize()} Accuracy: {epoch_accuracy:.4f}")

#         if phase == 'val' and epoch_loss < best_val_loss:
#             best_val_loss = epoch_loss
#             torch.save(model.state_dict(), "best_model.pth")
#             print("Saved new best model!")

#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Current learning rate: {current_lr}")

# print("Training completed!")



#########################       
import time
import torch
from tqdm import tqdm
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch

if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Currently using: GPU {torch.cuda.current_device()}")
else:
    print("CUDA is not available, using CPU")
  # Print GPU or CPU statusnum_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_accuracy = 0.0
        epoch_start_time = time.time()

        dataloader = train_loader if phase == 'train' else val_loader
        total_batches = len(dataloader)

        pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"{phase.capitalize()} Epoch {epoch+1}")
        
        for batch, (images, masks) in pbar:
            batch_start_time = time.time()  # Start time for each batch
            images, masks = images.to(device), masks.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(images).logits
                loss = criterion(outputs, masks)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            acc = calculate_accuracy(preds, masks)

            # Track metrics
            running_loss += loss.item() * images.size(0)
            running_accuracy += acc * images.size(0)

            # Compute metrics per step
            avg_loss = running_loss / ((batch + 1) * images.size(0))
            avg_acc = running_accuracy / ((batch + 1) * images.size(0))
            time_per_step = time.time() - batch_start_time  # Compute batch time

            # Update progress bar with additional details
            pbar.set_postfix({
                'Step': f'{batch+1}/{total_batches}',
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Loss': f'{avg_loss:.4f}',
                'Batch Acc': f'{acc:.4f}',
                'Mean Acc': f'{avg_acc:.4f}',
                'Time/Step (s)': f'{time_per_step:.2f}'
            })

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = running_accuracy / len(dataloader.dataset)
        epoch_time = time.time() - epoch_start_time

        print(f"\n{phase.capitalize()} Epoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f} sec, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")

print("\nTraining completed!")

############################################################################################################
# Inference Function
def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).logits
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    return prediction
