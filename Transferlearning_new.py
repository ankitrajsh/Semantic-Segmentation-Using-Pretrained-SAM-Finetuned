import os
import torch
import numpy as np
import random
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from tqdm import tqdm
import cv2
from PIL import Image

# 🚀 Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📂 Define Paths
IMAGE_DIR = "C:/Users/Nishith/Downloads/New_train_Patches"
MASK_DIR = "G:/New_VOC_Label_Patches"


# 🔹 Image Transformations
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor()
])


# 🔹 Custom Dataset for Semantic Segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        
        # Store mapping of filename to classes in the mask
        self.file_class_map = self._compute_class_presence()

    def _compute_class_presence(self):
        file_class_map = {}
        for filename in self.image_filenames:
            mask_path = os.path.join(self.mask_dir, filename.replace('.jpg', '.png'))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            unique_classes = np.unique(mask)
            file_class_map[filename] = unique_classes
        return file_class_map

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = image_name.replace('.jpg', '.png')

        # Load Image & Mask
        image = cv2.imread(os.path.join(self.image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        # Ensure mask values are within [0, 1, 2, 3]
        mask = np.clip(mask, 0, 3)

        # Debugging Check
        if 1 not in np.unique(mask) or 2 not in np.unique(mask):
            print(f"Warning: Missing classes in {mask_name} - Unique values: {np.unique(mask)}")

        # Apply Transformations
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.long)  # Ensure correct dtype

        return image, mask

# 🔹 Custom Sampler for Balanced Batches
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_class = {1: [], 2: [], 3: []}

        # Group indices by present class labels
        for idx, filename in enumerate(dataset.file_class_map.keys()):
            classes = dataset.file_class_map[filename]
            for c in classes:
                if c in self.indices_by_class:
                    self.indices_by_class[c].append(idx)

        # Flatten indices list
        self.all_indices = list(range(len(dataset)))

    def __iter__(self):
        batch = []
        while len(self.all_indices) > 0:
            # Ensure at least one patch from each class
            for c in [1, 2, 3]:
                if len(self.indices_by_class[c]) > 0:
                    batch.append(self.indices_by_class[c].pop())

            # Fill the rest of the batch randomly
            while len(batch) < self.batch_size and len(self.all_indices) > 0:
                batch.append(self.all_indices.pop())

            yield batch
            batch = []

    def __len__(self):
        return len(self.dataset) // self.batch_size

# 🏋️ Load Dataset
dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transform)
batch_size = 16
train_loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(dataset, batch_size))

# 🧠 Load Pretrained SegFormer Model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
image_processor = AutoImageProcessor.from_pretrained(model_name)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=4,
    ignore_mismatched_sizes=True
).to(device)

# ⚙️ Optimizer & Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# 🔄 Training Function
def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for images, masks in tqdm(train_loader, desc="Training"):
        # Debugging Info
        print("Before Moving to GPU:")
        print(f"Mask dtype: {masks.dtype}")
        print(f"Unique mask values: {torch.unique(masks)}")

        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(pixel_values=images).logits  # Shape: (batch, num_classes, H, W)
        outputs = F.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)

        # Compute Loss
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Compute Accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

    accuracy = correct / total
    return epoch_loss / len(train_loader), accuracy

# 🔬 Validation Function
def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)

            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()

    accuracy = correct / total
    return val_loss / len(val_loader), accuracy

# 🚀 Train the Model
num_epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# 🎯 Save Model
torch.save(model.state_dict(), "segformer_trained.pth")
print("Model saved!")
