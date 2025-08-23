"""
Simple Multi-Class Training Script for SAM-based Prostate Segmentation
=====================================================================

This script demonstrates how to train SAM for multi-class semantic segmentation
where each pixel gets assigned a class value (0-5) instead of binary masks.

Usage:
    python train_multiclass_sam.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import tifffile as tiff  # For TIFF support in medical imaging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

# Import your dataset class
from semantic_segmentation_multiclass import MultiClassSAMDataset, MultiClassMetrics


class SimplifiedMultiClassSAM(nn.Module):
    """Simplified SAM model for multi-class segmentation."""
    
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Load pretrained SAM
        self.sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.num_classes = num_classes
        
        # Freeze encoders (only train mask decoder)
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        # Modify the final layer of mask decoder for multi-class output
        self._modify_mask_decoder()
        
        print(f"Model initialized for {num_classes} classes")
        print("Encoders frozen - only mask decoder will be trained")
    
    def _modify_mask_decoder(self):
        """Modify mask decoder to output multiple classes."""
        # The mask decoder's final layer is in output_upscaling
        # We need to change it from 1 channel (binary) to num_classes channels
        
        # Access the output upscaling layers
        upscaling = self.sam.mask_decoder.output_upscaling
        
        # Find the final conv layer and modify it
        for i, layer in enumerate(upscaling):
            if isinstance(layer, nn.Conv2d):
                if i == len(upscaling) - 1:  # Last conv layer
                    # Get input channels from the layer
                    in_channels = layer.in_channels
                    
                    # Create new layer with same config but different output channels
                    new_layer = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.num_classes,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=layer.bias is not None
                    )
                    
                    # Initialize the new layer
                    nn.init.kaiming_normal_(new_layer.weight)
                    if new_layer.bias is not None:
                        nn.init.zeros_(new_layer.bias)
                    
                    # Replace the layer
                    upscaling[i] = new_layer
                    print(f"Modified final layer: {in_channels} -> {self.num_classes} channels")
                    break
    
    def forward(self, pixel_values, input_boxes):
        """Forward pass through the model."""
        outputs = self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )
        return outputs


def train_multiclass_sam():
    """Main training function."""
    
    # Configuration
    CONFIG = {
        'image_dir': "data/images",
        'mask_dir': "data/masks", 
        'batch_size': 2,
        'num_epochs': 5,
        'learning_rate': 1e-5,
        'num_classes': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': 'multiclass_sam_model.pth'
    }
    
    print(f"Configuration: {CONFIG}")
    print(f"Using device: {CONFIG['device']}")
    
    # Class information
    CLASS_NAMES = {
        0: "Background",
        1: "Benign", 
        2: "Gleason_3",
        3: "Gleason_4", 
        4: "Gleason_5",
        5: "Stroma"
    }
    
    # Initialize processor and dataset
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Create dataset
    dataset = MultiClassSAMDataset(
        image_dir=CONFIG['image_dir'],
        mask_dir=CONFIG['mask_dir'],
        processor=processor,
        validate_classes=True
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    # Initialize model
    model = SimplifiedMultiClassSAM(num_classes=CONFIG['num_classes'])
    model.to(CONFIG['device'])
    
    # Loss function with class weights (background gets lower weight)
    class_weights = torch.tensor([0.1, 1.0, 2.0, 2.5, 3.0, 1.5]).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (only optimize mask decoder)
    optimizer = torch.optim.Adam(
        model.sam.mask_decoder.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-4
    )
    
    # Metrics tracker
    metrics = MultiClassMetrics(num_classes=CONFIG['num_classes'])
    
    # Training loop
    print("\\nStarting training...")
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': [], 'dice': [], 'iou': []}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            pixel_values = batch["pixel_values"].to(CONFIG['device'])
            input_boxes = batch["input_boxes"].to(CONFIG['device'])
            ground_truth = batch["ground_truth_mask"].to(CONFIG['device'])
            
            # Ensure ground truth is proper shape and type
            if ground_truth.dim() == 4:
                ground_truth = ground_truth.squeeze(1)  # Remove channel dim if present
            ground_truth = ground_truth.long()  # Ensure long type for CrossEntropyLoss
            
            # Forward pass
            outputs = model(pixel_values, input_boxes)
            logits = outputs.pred_masks  # [B, num_classes, H, W]
            
            # Handle dimension mismatch
            if logits.dim() == 5:
                logits = logits.squeeze(1)  # Remove extra dimension if present
            
            # Calculate loss
            loss = criterion(logits, ground_truth)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predicted = torch.argmax(logits, dim=1)
                
                # Calculate batch metrics
                batch_accuracy = torch.mean((predicted == ground_truth).float()).item()
                batch_metrics = metrics.calculate_batch_metrics(predicted, ground_truth)
                
                epoch_metrics['accuracy'].append(batch_accuracy)
                epoch_metrics['dice'].append(batch_metrics['dice'])
                epoch_metrics['iou'].append(batch_metrics['iou'])
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{batch_accuracy:.3f}",
                'Dice': f"{batch_metrics['dice']:.3f}"
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = np.mean(epoch_metrics['accuracy'])
        avg_dice = np.mean(epoch_metrics['dice'])
        avg_iou = np.mean(epoch_metrics['iou'])
        
        print(f"\\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Average Dice: {avg_dice:.4f}")
        print(f"  Average IoU: {avg_iou:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': {
                    'accuracy': avg_accuracy,
                    'dice': avg_dice,
                    'iou': avg_iou
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), CONFIG['save_path'])
    print(f"\\nTraining completed! Model saved to: {CONFIG['save_path']}")
    
    return model


def test_model_predictions(model, dataset, device, num_samples=3):
    """Test the model and visualize predictions."""
    
    model.eval()
    
    # Define colors for visualization
    colors = np.array([
        [0, 0, 0],        # Background - Black
        [0, 0, 255],      # Benign - Blue
        [0, 255, 0],      # Gleason 3 - Green
        [255, 255, 0],    # Gleason 4 - Yellow
        [255, 0, 0],      # Gleason 5 - Red
        [128, 0, 128],    # Stroma - Purple
    ]) / 255.0  # Normalize to [0, 1]
    
    class_names = {
        0: "Background", 1: "Benign", 2: "Gleason_3",
        3: "Gleason_4", 4: "Gleason_5", 5: "Stroma"
    }
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            
            # Prepare inputs
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            input_boxes = sample["input_boxes"].unsqueeze(0).to(device)
            ground_truth = sample["ground_truth_mask"].numpy()
            
            # Get prediction
            outputs = model(pixel_values, input_boxes)
            logits = outputs.pred_masks.squeeze(0)  # Remove batch dim
            
            if logits.dim() == 4:
                logits = logits.squeeze(0)  # Remove extra dim if present
            
            predicted = torch.argmax(logits, dim=0).cpu().numpy()
            
            # Original image
            original_img = sample["pixel_values"].permute(1, 2, 0).numpy()
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f"Original Image\\n{sample['image_file']}")
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_colored = colors[ground_truth]
            axes[i, 1].imshow(gt_colored)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Prediction
            pred_colored = colors[predicted]
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            # Print class distribution
            unique_gt, counts_gt = np.unique(ground_truth, return_counts=True)
            unique_pred, counts_pred = np.unique(predicted, return_counts=True)
            
            print(f"\\nSample {i+1}:")
            print("Ground Truth classes:", [(class_names[c], cnt) for c, cnt in zip(unique_gt, counts_gt)])
            print("Predicted classes:", [(class_names[c], cnt) for c, cnt in zip(unique_pred, counts_pred)])
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\nVisualization saved as 'prediction_results.png'")


if __name__ == "__main__":
    # Check if data directories exist
    if not os.path.exists("data/images") or not os.path.exists("data/masks"):
        print("Error: Please create 'data/images' and 'data/masks' directories with your training data")
        print("Make sure your masks contain class values 0-5 for proper multi-class segmentation")
        exit(1)
    
    # Train the model
    print("=== Multi-Class SAM Training ===")
    print("Classes:")
    print("  0: Background (Black)")
    print("  1: Benign (Blue)")
    print("  2: Gleason Pattern 3 (Green)")
    print("  3: Gleason Pattern 4 (Yellow)")
    print("  4: Gleason Pattern 5 (Red)")
    print("  5: Stroma (Purple)")
    print()
    
    trained_model = train_multiclass_sam()
    
    # Test the model
    print("\\n=== Testing Model Predictions ===")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = MultiClassSAMDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        processor=processor
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_model_predictions(trained_model, test_dataset, device)
