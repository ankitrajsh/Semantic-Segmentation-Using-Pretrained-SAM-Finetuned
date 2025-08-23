#!/usr/bin/env python3
"""
Prostate Cancer Semantic Segmentation Workflow
==============================================

This script provides a comprehensive workflow for training and evaluating
a fine-tuned SAM model for prostate cancer Gleason grading.

Author: Ankit Sharma
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from monai.losses import DiceCELoss
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the segmentation workflow"""
    
    # Data paths
    DATA_DIR = "data"
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    MASK_DIR = os.path.join(DATA_DIR, "masks")
    
    # Model configuration
    MODEL_NAME = "facebook/sam-vit-base"
    
    # Training parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 4
    LEARNING_RATE = 5e-5
    
    # Image parameters
    PATCH_SIZE = 256
    
    # Classes for Gleason grading
    CLASS_NAMES = {
        0: "Background",
        1: "Benign",
        2: "Gleason_3", 
        3: "Gleason_4",
        4: "Gleason_5",
        5: "Stroma"
    }
    
    CLASS_COLORS = {
        0: "black",
        1: "blue",
        2: "green",
        3: "yellow", 
        4: "red",
        5: "purple"
    }

def setup_environment():
    """Setup the environment and check dependencies"""
    logger.info("Setting up environment...")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("GPU not available, using CPU")
    
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.IMAGE_DIR, exist_ok=True)
    os.makedirs(Config.MASK_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    return device

def check_data_availability():
    """Check if training data is available"""
    logger.info("Checking data availability...")
    
    if not os.path.exists(Config.IMAGE_DIR) or not os.path.exists(Config.MASK_DIR):
        logger.error("Data directories not found!")
        return False
    
    image_files = [f for f in os.listdir(Config.IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(Config.MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    if len(image_files) == 0 or len(mask_files) == 0:
        logger.error("No training data found!")
        return False
    
    if len(image_files) != len(mask_files):
        logger.warning(f"Mismatch in number of images ({len(image_files)}) and masks ({len(mask_files)})")
    
    return True

def preprocess_data():
    """Preprocess data if needed"""
    logger.info("Preprocessing data...")
    
    # This function can be expanded to include:
    # - Patch extraction from large images
    # - Data augmentation
    # - Quality checks
    # - Data splitting
    
    # For now, assume data is already preprocessed
    logger.info("Data preprocessing completed")

def create_dataset_and_dataloader():
    """Create dataset and dataloader"""
    from datasets import Dataset
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset as TorchDataset
    
    logger.info("Creating dataset and dataloader...")
    
    # Load image and mask files
    image_files = sorted([f for f in os.listdir(Config.IMAGE_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(Config.MASK_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def load_images(files, directory):
        images = []
        for file in files:
            try:
                with Image.open(os.path.join(directory, file)) as img:
                    images.append(img.copy())
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        return images
    
    # Create dataset dictionary
    dataset_dict = {
        "image": load_images(image_files, Config.IMAGE_DIR),
        "label": load_images(mask_files, Config.MASK_DIR)
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create SAM dataset
    class SAMDataset(TorchDataset):
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
            
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = ground_truth_mask
            return inputs
        
        def get_bounding_box(self, ground_truth_map):
            y_indices, x_indices = np.where(ground_truth_map > 0)
            if y_indices.size == 0 or x_indices.size == 0:
                return [0, 0, 1, 1]
            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Add small perturbation
            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            
            return [x_min, y_min, x_max, y_max]
    
    # Initialize processor and create dataset
    processor = SamProcessor.from_pretrained(Config.MODEL_NAME)
    sam_dataset = SAMDataset(dataset, processor)
    
    # Create dataloader
    dataloader = DataLoader(sam_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    logger.info(f"Created dataset with {len(sam_dataset)} samples")
    return dataloader, processor

def create_model(device):
    """Create and initialize the SAM model"""
    logger.info("Creating SAM model...")
    
    model = SamModel.from_pretrained(Config.MODEL_NAME)
    model.to(device)
    
    # Freeze some parameters if needed
    # for param in model.vision_encoder.parameters():
    #     param.requires_grad = False
    
    logger.info("Model created and moved to device")
    return model

def calculate_metrics(pred_masks, gt_masks):
    """Calculate segmentation metrics"""
    def dice_coefficient(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def iou_score(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred) > 0.5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)
    
    dice = dice_coefficient(pred_masks, gt_masks)
    iou = iou_score(pred_masks, gt_masks)
    
    return dice.item(), iou.item()

def train_model(model, dataloader, device, num_epochs=None):
    """Train the SAM model"""
    if num_epochs is None:
        num_epochs = Config.NUM_EPOCHS
        
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = DiceCELoss()
    
    model.train()
    
    training_history = {
        'epoch_losses': [],
        'epoch_dice': [],
        'epoch_iou': []
    }
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_dice = []
        epoch_iou = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            try:
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_boxes=batch["input_boxes"].to(device),
                    multimask_output=False
                )
                
                # Compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
                
                # Calculate metrics
                dice, iou = calculate_metrics(predicted_masks, ground_truth_masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Store metrics
                epoch_losses.append(loss.item())
                epoch_dice.append(dice)
                epoch_iou.append(iou)
                
                # Update progress bar
                progress_bar.set_description(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Dice: {dice:.4f}, "
                    f"IoU: {iou:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_dice = np.mean(epoch_dice) if epoch_dice else 0
        avg_iou = np.mean(epoch_iou) if epoch_iou else 0
        
        training_history['epoch_losses'].append(avg_loss)
        training_history['epoch_dice'].append(avg_dice)
        training_history['epoch_iou'].append(avg_iou)
        
        logger.info(
            f'Epoch {epoch+1}/{num_epochs} - '
            f'Avg Loss: {avg_loss:.4f}, '
            f'Avg Dice: {avg_dice:.4f}, '
            f'Avg IoU: {avg_iou:.4f}'
        )
    
    return model, training_history

def save_model(model, save_path="models/sam_prostate_segmentation.pth"):
    """Save the trained model"""
    logger.info(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved successfully")

def plot_training_history(history, save_path="results/training_history.png"):
    """Plot training metrics"""
    logger.info("Plotting training history...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['epoch_losses'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Plot Dice coefficient
    axes[1].plot(history['epoch_dice'])
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].grid(True)
    
    # Plot IoU
    axes[2].plot(history['epoch_iou'])
    axes[2].set_title('IoU Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Training history saved to {save_path}")

def visualize_predictions(model, dataloader, device, num_samples=3):
    """Visualize model predictions"""
    logger.info("Visualizing predictions...")
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )
            
            predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
            
            # Plot results
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for j in range(min(3, batch["pixel_values"].shape[0])):
                # Original image
                img = batch["pixel_values"][j].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                axes[0, j].imshow(img)
                axes[0, j].set_title(f'Input Image {j+1}')
                axes[0, j].axis('off')
                
                # Ground truth vs prediction
                gt_mask = batch["ground_truth_mask"][j].cpu().numpy()
                pred_mask = predicted_masks[j].cpu().numpy()
                
                axes[1, j].imshow(gt_mask, alpha=0.7, cmap='viridis')
                axes[1, j].imshow(pred_mask > 0.5, alpha=0.5, cmap='Reds')
                axes[1, j].set_title(f'GT (green) vs Pred (red) {j+1}')
                axes[1, j].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/predictions_batch_{i}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main workflow function"""
    parser = argparse.ArgumentParser(description='Prostate Cancer Segmentation Workflow')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and load existing model')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    logger.info("Starting Prostate Cancer Semantic Segmentation Workflow")
    logger.info("=" * 60)
    
    # Setup environment
    device = setup_environment()
    
    # Check data availability
    if not check_data_availability():
        logger.error("Please ensure training data is available in the data directory")
        return
    
    # Preprocess data
    preprocess_data()
    
    # Create dataset and dataloader
    try:
        dataloader, processor = create_dataset_and_dataloader()
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return
    
    # Create model
    try:
        model = create_model(device)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return
    
    if not args.skip_training:
        # Train model
        try:
            model, history = train_model(model, dataloader, device, args.epochs)
            
            # Save model
            save_model(model)
            
            # Plot training history
            plot_training_history(history)
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return
    else:
        logger.info("Skipping training, loading existing model...")
        try:
            model.load_state_dict(torch.load("models/sam_prostate_segmentation.pth"))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    # Visualize predictions
    try:
        visualize_predictions(model, dataloader, device)
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
    
    logger.info("Workflow completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
