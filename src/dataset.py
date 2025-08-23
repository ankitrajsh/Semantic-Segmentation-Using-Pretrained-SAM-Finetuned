   """
Dataset classes for prostate cancer semantic segmentation.

This module contains dataset classes optimized for SAM-based segmentation
of prostate histopathology images with Gleason grading.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict, Union
import random
from transformers import SamProcessor


class SAMProstateDataset(Dataset):
    """
    Dataset class for SAM-based prostate cancer segmentation.
    
    This dataset is specifically designed for prostate histopathology images
    with Gleason grading annotations.
    
    Args:
        image_dir (str): Directory containing image patches
        mask_dir (str): Directory containing corresponding mask patches
        processor: SAM processor for input preprocessing
        transform: Optional data transforms
        class_mapping (dict): Mapping from pixel values to class indices
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        processor: SamProcessor,
        transform=None,
        class_mapping: Dict[int, int] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform
        
        # Default class mapping for Gleason grading
        if class_mapping is None:
            self.class_mapping = {
                0: 0,    # Background
                1: 1,    # Benign (Blue)
                2: 2,    # Gleason 3 (Green)
                3: 3,    # Gleason 4 (Yellow)
                4: 4,    # Gleason 5 (Red)
                5: 5     # Stroma (Purple)
            }
        else:
            self.class_mapping = class_mapping
        
        # Get file lists
        self.image_files = self._get_image_files()
        self.mask_files = self._get_mask_files()
        
        # Validate file correspondence
        self._validate_files()
    
    def _get_image_files(self) -> List[str]:
        """Get sorted list of image files."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        files = [f for f in os.listdir(self.image_dir) 
                if f.lower().endswith(valid_extensions)]
        return sorted(files)
    
    def _get_mask_files(self) -> List[str]:
        """Get sorted list of mask files."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        files = [f for f in os.listdir(self.mask_dir) 
                if f.lower().endswith(valid_extensions)]
        return sorted(files)
    
    def _validate_files(self):
        """Validate that images and masks correspond correctly."""
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) doesn't match "
                f"number of masks ({len(self.mask_files)})"
            )
        
        # Check if filenames correspond (basic check)
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_base = os.path.splitext(img_file)[0]
            mask_base = os.path.splitext(mask_file)[0]
            if img_base != mask_base:
                print(f"Warning: {img_file} may not correspond to {mask_file}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict containing processed inputs for SAM model
        """
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Convert to numpy
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed['image']
            mask_np = transformed['mask']
            image = Image.fromarray(image_np)
        
        # Generate bounding box prompt
        bbox = self._get_bounding_box(mask_np)
        
        # Process with SAM processor
        inputs = self.processor(
            image, 
            input_boxes=[[bbox]], 
            return_tensors="pt"
        )
        
        # Remove batch dimension and add ground truth
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np, dtype=torch.float32)
        inputs["file_name"] = self.image_files[idx]
        
        return inputs
    
    def _get_bounding_box(self, mask: np.ndarray) -> List[int]:
        """
        Generate bounding box from mask for SAM prompting.
        
        Args:
            mask (np.ndarray): Ground truth mask
            
        Returns:
            List[int]: Bounding box coordinates [x_min, y_min, x_max, y_max]
        """
        # Find non-zero pixels
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Return small default box if no foreground
            return [0, 0, 10, 10]
        
        # Get bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add small random perturbation to improve robustness
        H, W = mask.shape
        perturbation = 5
        
        x_min = max(0, x_min - random.randint(0, perturbation))
        x_max = min(W - 1, x_max + random.randint(0, perturbation))
        y_min = max(0, y_min - random.randint(0, perturbation))
        y_max = min(H - 1, y_max + random.randint(0, perturbation))
        
        return [x_min, y_min, x_max, y_max]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Analyze class distribution in the dataset.
        
        Returns:
            Dict with class names and pixel counts
        """
        class_counts = {}
        class_names = {
            0: "Background",
            1: "Benign", 
            2: "Gleason_3",
            3: "Gleason_4", 
            4: "Gleason_5",
            5: "Stroma"
        }
        
        for class_id, class_name in class_names.items():
            class_counts[class_name] = 0
        
        print("Analyzing class distribution...")
        for i in range(len(self)):
            mask_path = os.path.join(self.mask_dir, self.mask_files[i])
            mask = np.array(Image.open(mask_path).convert('L'))
            
            unique_classes, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique_classes, counts):
                if class_id in class_names:
                    class_counts[class_names[class_id]] += count
        
        return class_counts
    
    def visualize_sample(self, idx: int, save_path: str = None):
        """
        Visualize a sample from the dataset.
        
        Args:
            idx (int): Index of sample to visualize
            save_path (str): Optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        # Get original files
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title(f'Image: {self.image_files[idx]}')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title(f'Mask: {self.mask_files[idx]}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ProstateDatasetFromDict(Dataset):
    """
    Dataset class that works with HuggingFace datasets format.
    
    This is compatible with the existing Main_final_SAM.py implementation.
    """
    
    def __init__(self, dataset_dict: Dict, processor: SamProcessor):
        self.dataset = dataset_dict
        self.processor = processor
    
    def __len__(self) -> int:
        return len(self.dataset['image'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        
        # Generate bounding box
        bbox = self._get_bounding_box(ground_truth_mask)
        
        # Process with SAM
        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        
        return inputs
    
    def _get_bounding_box(self, ground_truth_map: np.ndarray) -> List[int]:
        """Generate bounding box from ground truth mask."""
        y_indices, x_indices = np.where(ground_truth_map > 0)
        
        if y_indices.size == 0 or x_indices.size == 0:
            return [0, 0, 1, 1]
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add perturbation
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        
        return [x_min, y_min, x_max, y_max]


def create_data_loaders(
    image_dir: str,
    mask_dir: str,
    processor: SamProcessor,
    batch_size: int = 2,
    train_split: float = 0.8,
    shuffle: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        image_dir: Directory with training images
        mask_dir: Directory with training masks
        processor: SAM processor
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = SAMProstateDataset(image_dir, mask_dir, processor)
    
    # Split into train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def analyze_dataset_statistics(image_dir: str, mask_dir: str):
    """
    Analyze and print dataset statistics.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
    """
    # Create a dummy processor for analysis
    from transformers import SamProcessor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    dataset = SAMProstateDataset(image_dir, mask_dir, processor)
    
    print(f"Dataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    # Analyze class distribution
    class_dist = dataset.get_class_distribution()
    print(f"\nClass Distribution:")
    for class_name, count in class_dist.items():
        percentage = count / sum(class_dist.values()) * 100
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
    
    # Analyze image sizes
    print(f"\nAnalyzing image dimensions...")
    widths, heights = [], []
    
    for i in range(min(100, len(dataset))):  # Sample first 100 images
        img_path = os.path.join(image_dir, dataset.image_files[i])
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    
    print(f"Image dimensions (sampled {len(widths)} images):")
    print(f"  Width: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")
    print(f"  Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")


if __name__ == "__main__":
    # Example usage
    image_dir = "data/images"
    mask_dir = "data/masks"
    
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        analyze_dataset_statistics(image_dir, mask_dir)
    else:
        print("Please ensure data directories exist with images and masks")