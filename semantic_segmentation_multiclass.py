"""
Multi-Class Semantic Segmentation for Prostate Cancer using SAM
===============================================================

This module implements proper multi-class semantic segmentation where each pixel
is assigned a class value (0-5) for prostate cancer Gleason grading.

Classes:
    0: Background (Black)
    1: Benign (Blue)  
    2: Gleason Pattern 3 (Green)
    3: Gleason Pattern 4 (Yellow)
    4: Gleason Pattern 5 (Red)
    5: Stroma (Purple)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import tifffile as tiff  # For TIFF support
from transformers import SamModel, SamProcessor
from monai.losses import DiceCELoss
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiClassSAMDataset(Dataset):
    """
    Dataset class for multi-class semantic segmentation.
    
    Each mask should contain integer values 0-5 representing different classes.
    """
    
    def __init__(self, image_dir: str, mask_dir: str, processor: SamProcessor, 
                 transforms=None, validate_classes=True):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing segmentation masks
            processor: SAM processor for image preprocessing
            transforms: Optional data augmentations
            validate_classes: Whether to validate class values in masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transforms = transforms
        
        # Get sorted file lists - support multiple formats including TIFF
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        mask_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(image_extensions)])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) 
                                if f.lower().endswith(mask_extensions)])
        
        # Validate file correspondence
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        # Class information
        self.num_classes = 6
        self.class_names = {
            0: "Background",
            1: "Benign", 
            2: "Gleason_3",
            3: "Gleason_4", 
            4: "Gleason_5",
            5: "Stroma"
        }
        
        # Validate masks if requested
        if validate_classes:
            self._validate_mask_classes()
        
        logger.info(f"Loaded {len(self.image_files)} image-mask pairs")
        logger.info(f"Image formats found: {set(f.split('.')[-1].lower() for f in self.image_files)}")
        logger.info(f"Mask formats found: {set(f.split('.')[-1].lower() for f in self.mask_files)}")
    
    def _validate_mask_classes(self):
        """Validate that masks contain only valid class values (0-5)."""
        logger.info("Validating mask class values...")
        
        invalid_files = []
        for mask_file in self.mask_files[:5]:  # Check first 5 files
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            
            # Handle RGB masks (convert to single channel)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take first channel
            
            unique_values = np.unique(mask)
            invalid_values = unique_values[unique_values > 5]
            
            if len(invalid_values) > 0:
                invalid_files.append((mask_file, invalid_values))
        
        if invalid_files:
            logger.warning(f"Found invalid class values in masks: {invalid_files}")
        else:
            logger.info("All validated masks contain valid class values (0-5)")
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_image(self, file_path: str):
        """Load image with support for multiple formats including TIFF."""
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext in ['tiff', 'tif']:
            # Use tifffile for TIFF images (better for medical imaging)
            image_array = tiff.imread(file_path)
            
            # Handle different TIFF formats
            if len(image_array.shape) == 2:
                # Grayscale TIFF - convert to RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[0] == 3:
                # Channel-first RGB TIFF - convert to channel-last
                image_array = np.transpose(image_array, (1, 2, 0))
            elif len(image_array.shape) == 3 and image_array.shape[-1] > 3:
                # Multi-channel TIFF - take first 3 channels
                image_array = image_array[:, :, :3]
            
            # Normalize to 0-255 if needed
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            return Image.fromarray(image_array)
        else:
            # Use PIL for standard formats
            return Image.open(file_path).convert("RGB")
    
    def _load_mask(self, file_path: str):
        """Load mask with support for multiple formats including TIFF."""
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext in ['tiff', 'tif']:
            # Use tifffile for TIFF masks
            mask_array = tiff.imread(file_path)
            
            # Handle different TIFF mask formats
            if len(mask_array.shape) == 3:
                # Multi-channel TIFF - take first channel or convert RGB to class indices
                if mask_array.shape[-1] == 3:
                    # RGB mask - convert to class indices
                    mask_array = self._rgb_to_class_mask(mask_array)
                else:
                    # Take first channel
                    mask_array = mask_array[:, :, 0]
            
            return mask_array
        else:
            # Use PIL for standard formats
            mask = Image.open(file_path)
            mask_array = np.array(mask)
            
            # Handle RGB masks
            if len(mask_array.shape) == 3:
                mask_array = self._rgb_to_class_mask(mask_array)
            
            return mask_array
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing processed image, mask, and metadata
        """
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load using format-specific loaders
        image = self._load_image(image_path)
        mask_array = self._load_mask(mask_path)
        
        # Handle RGB masks (convert to class indices)
        if len(mask_array.shape) == 3:
            mask_array = self._rgb_to_class_mask(mask_array)
        
        # Ensure mask values are in valid range
        mask_array = np.clip(mask_array, 0, self.num_classes - 1)
        
        # Generate bounding box for SAM prompt
        bbox = self._get_bounding_box(mask_array)
        
        # Process image with SAM processor
        inputs = self.processor(
            image, 
            input_boxes=[[bbox]], 
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add ground truth mask
        inputs["ground_truth_mask"] = torch.tensor(mask_array, dtype=torch.long)
        inputs["image_file"] = self.image_files[idx]
        inputs["mask_file"] = self.mask_files[idx]
        
        return inputs
    
    def _rgb_to_class_mask(self, rgb_mask: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class indices.
        
        This function maps RGB colors to class indices based on
        the predefined color scheme.
        """
        # Define color to class mapping
        color_to_class = {
            (0, 0, 0): 0,        # Background - Black
            (0, 0, 255): 1,      # Benign - Blue
            (0, 255, 0): 2,      # Gleason 3 - Green
            (255, 255, 0): 3,    # Gleason 4 - Yellow
            (255, 0, 0): 4,      # Gleason 5 - Red
            (128, 0, 128): 5,    # Stroma - Purple
        }
        
        # Initialize class mask
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        
        # Map colors to classes
        for color, class_id in color_to_class.items():
            mask = np.all(rgb_mask == color, axis=2)
            class_mask[mask] = class_id
        
        return class_mask
    
    def _get_bounding_box(self, mask: np.ndarray) -> List[int]:
        """
        Generate bounding box from mask for SAM prompt.
        
        Args:
            mask: Segmentation mask with class indices
            
        Returns:
            Bounding box coordinates [x_min, y_min, x_max, y_max]
        """
        # Find all non-background pixels
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            # Return small box at origin if no foreground pixels
            return [0, 0, 10, 10]
        
        # Calculate bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add random perturbation for data augmentation
        H, W = mask.shape
        perturbation = np.random.randint(5, 25)
        
        x_min = max(0, x_min - perturbation)
        x_max = min(W - 1, x_max + perturbation)
        y_min = max(0, y_min - perturbation)
        y_max = min(H - 1, y_max + perturbation)
        
        return [x_min, y_min, x_max, y_max]


class MultiClassSAMModel(nn.Module):
    """
    SAM model adapted for multi-class semantic segmentation.
    
    This model modifies SAM's mask decoder to output multiple classes
    instead of binary masks.
    """
    
    def __init__(self, model_name: str = "facebook/sam-vit-base", 
                 num_classes: int = 6, freeze_encoder: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.sam = SamModel.from_pretrained(model_name)
        
        # Freeze encoder components if specified
        if freeze_encoder:
            self._freeze_encoders()
        
        # Modify mask decoder for multi-class output
        self._modify_mask_decoder()
    
    def _freeze_encoders(self):
        """Freeze vision encoder and prompt encoder."""
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
        
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        logger.info("Froze vision encoder and prompt encoder")
    
    def _modify_mask_decoder(self):
        """Modify mask decoder to output multiple classes."""
        # Get the original output projection layer
        original_output_dim = self.sam.mask_decoder.output_upscaling[3].out_channels
        
        # Replace the final layer to output num_classes channels
        self.sam.mask_decoder.output_upscaling[3] = nn.Conv2d(
            original_output_dim, self.num_classes, kernel_size=1
        )
        
        logger.info(f"Modified mask decoder for {self.num_classes} classes")
    
    def forward(self, pixel_values, input_boxes, multimask_output=False):
        """Forward pass through the model."""
        outputs = self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )
        
        return outputs
    
    def predict_classes(self, pixel_values, input_boxes):
        """
        Predict class probabilities and final segmentation.
        
        Returns:
            Dict containing class probabilities and predicted segmentation
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_boxes)
            
            # Get class probabilities
            logits = outputs.pred_masks  # Shape: [B, num_classes, H, W]
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted classes
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predicted_classes': predicted_classes
            }


class MultiClassTrainer:
    """Training class for multi-class SAM segmentation."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Initialize loss function and metrics
        self.criterion = self._get_loss_function()
        self.metrics = MultiClassMetrics(num_classes=6)
    
    def _get_loss_function(self):
        """Get appropriate loss function for multi-class segmentation."""
        # Use class weights to handle imbalance
        class_weights = torch.tensor([0.1, 1.0, 2.0, 2.0, 3.0, 1.5]).to(self.device)
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'dice': [], 'iou': [], 'accuracy': []}
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_boxes = batch["input_boxes"].to(self.device)
            ground_truth = batch["ground_truth_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values, input_boxes)
            logits = outputs.pred_masks.squeeze(1)  # Remove singleton dim
            
            # Calculate loss
            loss = self.criterion(logits, ground_truth)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predicted = torch.argmax(logits, dim=1)
                batch_metrics = self.metrics.calculate_batch_metrics(
                    predicted, ground_truth
                )
                
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_metrics['dice']:.4f}",
                'acc': f"{batch_metrics['accuracy']:.4f}"
            })
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(dataloader)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                pixel_values = batch["pixel_values"].to(self.device)
                input_boxes = batch["input_boxes"].to(self.device)
                ground_truth = batch["ground_truth_mask"].to(self.device)
                
                outputs = self.model(pixel_values, input_boxes)
                logits = outputs.pred_masks.squeeze(1)
                
                loss = self.criterion(logits, ground_truth)
                val_loss += loss.item()
                
                predicted = torch.argmax(logits, dim=1)
                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(ground_truth.cpu().numpy())
        
        # Calculate comprehensive metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.metrics.calculate_comprehensive_metrics(predictions, targets)
        avg_loss = val_loss / len(dataloader)
        
        return avg_loss, metrics


class MultiClassMetrics:
    """Metrics calculation for multi-class segmentation."""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.class_names = {
            0: "Background", 1: "Benign", 2: "Gleason_3",
            3: "Gleason_4", 4: "Gleason_5", 5: "Stroma"
        }
    
    def calculate_batch_metrics(self, predictions, targets):
        """Calculate metrics for a single batch."""
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Calculate metrics
        dice = self._dice_coefficient(pred_np, target_np)
        iou = self._mean_iou(pred_np, target_np)
        accuracy = np.mean(pred_np == target_np)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy
        }
    
    def calculate_comprehensive_metrics(self, predictions, targets):
        """Calculate comprehensive metrics across all classes."""
        metrics = {}
        
        # Overall metrics
        metrics['overall_accuracy'] = np.mean(predictions == targets)
        metrics['mean_dice'] = self._dice_coefficient(predictions, targets)
        metrics['mean_iou'] = self._mean_iou(predictions, targets)
        
        # Per-class metrics
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            # Binary masks for this class
            pred_binary = (predictions == class_id).astype(np.float32)
            target_binary = (targets == class_id).astype(np.float32)
            
            if np.sum(target_binary) > 0:  # Only if class exists in targets
                dice = self._dice_coefficient(pred_binary, target_binary)
                iou = self._iou_score(pred_binary, target_binary)
                
                metrics[f'{class_name}_dice'] = dice
                metrics[f'{class_name}_iou'] = iou
        
        return metrics
    
    def _dice_coefficient(self, pred, target, smooth=1e-6):
        """Calculate Dice coefficient."""
        intersection = np.sum(pred * target)
        return (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    def _iou_score(self, pred, target, smooth=1e-6):
        """Calculate IoU score."""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def _mean_iou(self, predictions, targets):
        """Calculate mean IoU across all classes."""
        ious = []
        for class_id in range(self.num_classes):
            pred_binary = (predictions == class_id).astype(np.float32)
            target_binary = (targets == class_id).astype(np.float32)
            
            if np.sum(target_binary) > 0:
                iou = self._iou_score(pred_binary, target_binary)
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0


def visualize_predictions(images, targets, predictions, class_names, 
                         num_samples=4, figsize=(15, 10)):
    """
    Visualize segmentation predictions.
    
    Args:
        images: Input images
        targets: Ground truth masks
        predictions: Predicted masks
        class_names: Dictionary mapping class IDs to names
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    
    # Define colors for each class
    colors = np.array([
        [0, 0, 0],        # Background - Black
        [0, 0, 255],      # Benign - Blue
        [0, 255, 0],      # Gleason 3 - Green
        [255, 255, 0],    # Gleason 4 - Yellow
        [255, 0, 0],      # Gleason 5 - Red
        [128, 0, 128],    # Stroma - Purple
    ])
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_colored = colors[targets[i]]
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        pred_colored = colors[predictions[i]]
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training script for multi-class SAM segmentation."""
    # Configuration
    config = {
        'image_dir': "data/images",
        'mask_dir': "data/masks",
        'batch_size': 2,
        'num_epochs': 10,
        'learning_rate': 1e-5,
        'num_classes': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # Initialize processor and dataset
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    dataset = MultiClassSAMDataset(
        image_dir=config['image_dir'],
        mask_dir=config['mask_dir'],
        processor=processor
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Initialize model and trainer
    model = MultiClassSAMModel(num_classes=config['num_classes'])
    trainer = MultiClassTrainer(model, device=config['device'])
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.sam.mask_decoder.parameters(),
        lr=config['learning_rate']
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_loss, train_metrics = trainer.train_epoch(train_loader, optimizer)
        
        # Validation
        val_loss, val_metrics = trainer.validate(val_loader)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['mean_dice']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multiclass_sam_model.pth')
            logger.info("Saved new best model!")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
