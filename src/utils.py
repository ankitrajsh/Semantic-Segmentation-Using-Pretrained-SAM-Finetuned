   """
Utility functions for prostate cancer semantic segmentation.

This module provides helper functions for data processing, visualization,
and various utilities needed throughout the project.
"""

import os
import json
import pickle
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
from datetime import datetime
import logging


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    log_file: str = "training.log",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    return logger


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_directories(dirs: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {dir_path}")


def save_config(config: Dict, save_path: str):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {save_path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    save_path: str
):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    
    print(f"Checkpoint saved: {save_path}")


def load_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[int, float, Dict]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Tuple of (epoch, loss, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss, metrics


class ImageProcessor:
    """
    Image processing utilities for prostate histopathology images.
    """
    
    @staticmethod
    def normalize_image(image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize image using different methods.
        
        Args:
            image: Input image array
            method: Normalization method ('standard', 'minmax', 'zscore')
            
        Returns:
            Normalized image
        """
        if method == 'standard':
            # Normalize to [0, 1]
            return image.astype(np.float32) / 255.0
        elif method == 'minmax':
            # Min-max normalization
            return (image - image.min()) / (image.max() - image.min())
        elif method == 'zscore':
            # Z-score normalization
            return (image - image.mean()) / image.std()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def resize_image_and_mask(
        image: np.ndarray,
        mask: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image and mask while maintaining consistency.
        
        Args:
            image: Input image
            mask: Input mask
            target_size: Target (width, height)
            interpolation: Interpolation method
            
        Returns:
            Tuple of (resized_image, resized_mask)
        """
        resized_image = cv2.resize(image, target_size, interpolation=interpolation)
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        return resized_image, resized_mask
    
    @staticmethod
    def extract_patches(
        image: np.ndarray,
        patch_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None,
        padding: bool = False
    ) -> List[np.ndarray]:
        """
        Extract patches from large image.
        
        Args:
            image: Input image
            patch_size: Size of patches (height, width)
            stride: Stride for patch extraction
            padding: Whether to pad image for complete coverage
            
        Returns:
            List of image patches
        """
        if stride is None:
            stride = patch_size
        
        H, W = image.shape[:2]
        patch_h, patch_w = patch_size
        stride_h, stride_w = stride
        
        if padding:
            # Calculate padding needed
            pad_h = patch_h - (H % stride_h) if H % stride_h != 0 else 0
            pad_w = patch_w - (W % stride_w) if W % stride_w != 0 else 0
            
            if image.ndim == 3:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            H, W = image.shape[:2]
        
        patches = []
        for y in range(0, H - patch_h + 1, stride_h):
            for x in range(0, W - patch_w + 1, stride_w):
                patch = image[y:y+patch_h, x:x+patch_w]
                patches.append(patch)
        
        return patches
    
    @staticmethod
    def reconstruct_from_patches(
        patches: List[np.ndarray],
        original_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Reconstruct image from patches.
        
        Args:
            patches: List of patches
            original_size: Original image size (height, width)
            patch_size: Size of patches
            stride: Stride used for extraction
            
        Returns:
            Reconstructed image
        """
        if stride is None:
            stride = patch_size
        
        H, W = original_size
        patch_h, patch_w = patch_size
        stride_h, stride_w = stride
        
        # Determine if image has channels
        if patches[0].ndim == 3:
            channels = patches[0].shape[2]
            reconstructed = np.zeros((H, W, channels), dtype=patches[0].dtype)
            count_map = np.zeros((H, W, channels))
        else:
            reconstructed = np.zeros((H, W), dtype=patches[0].dtype)
            count_map = np.zeros((H, W))
        
        patch_idx = 0
        for y in range(0, H - patch_h + 1, stride_h):
            for x in range(0, W - patch_w + 1, stride_w):
                if patch_idx < len(patches):
                    reconstructed[y:y+patch_h, x:x+patch_w] += patches[patch_idx]
                    count_map[y:y+patch_h, x:x+patch_w] += 1
                    patch_idx += 1
        
        # Average overlapping regions
        count_map[count_map == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / count_map
        
        return reconstructed.astype(patches[0].dtype)


class Visualizer:
    """
    Visualization utilities for segmentation results.
    """
    
    def __init__(self, class_colors: Optional[Dict[int, str]] = None):
        if class_colors is None:
            self.class_colors = {
                0: '#000000',  # Background - Black
                1: '#0000FF',  # Benign - Blue
                2: '#00FF00',  # Gleason 3 - Green
                3: '#FFFF00',  # Gleason 4 - Yellow
                4: '#FF0000',  # Gleason 5 - Red
                5: '#800080'   # Stroma - Purple
            }
        else:
            self.class_colors = class_colors
    
    def visualize_sample(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: Optional[np.ndarray] = None,
        title: str = "Segmentation Results",
        save_path: Optional[str] = None
    ):
        """
        Visualize a single sample with ground truth and prediction.
        
        Args:
            image: Original image
            ground_truth: Ground truth mask
            prediction: Predicted mask (optional)
            title: Plot title
            save_path: Path to save the plot
        """
        if prediction is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(ground_truth, cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        if prediction is not None:
            axes[2].imshow(prediction, cmap='viridis')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of mask on image.
        
        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Transparency factor
            
        Returns:
            Overlay image
        """
        # Convert mask to color
        colored_mask = self.mask_to_color(mask)
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert class mask to color image.
        
        Args:
            mask: Class mask
            
        Returns:
            Colored mask
        """
        H, W = mask.shape
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_id, color_hex in self.class_colors.items():
            # Convert hex to RGB
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            colored_mask[mask == class_id] = color_rgb
        
        return colored_mask
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        Plot training history curves.
        
        Args:
            history: Dictionary with training metrics
            save_path: Path to save the plot
        """
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(history[metric])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def calculate_dataset_stats(dataloader: torch.utils.data.DataLoader) -> Dict:
    """
    Calculate dataset statistics.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Dictionary with dataset statistics
    """
    pixel_sum = torch.zeros(3)
    pixel_sum_sq = torch.zeros(3)
    total_pixels = 0
    
    for batch in dataloader:
        images = batch['pixel_values']
        batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        total_pixels += batch_pixels
        
        pixel_sum += images.sum(dim=[0, 2, 3])
        pixel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    
    mean = pixel_sum / total_pixels
    std = torch.sqrt(pixel_sum_sq / total_pixels - mean ** 2)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_pixels': total_pixels
    }


if __name__ == "__main__":
    print("Utility functions for prostate cancer segmentation")
    print("Available functions:")
    print("- set_random_seeds()")
    print("- get_device()")
    print("- ImageProcessor class")
    print("- Visualizer class")
    print("- save/load model checkpoints")
    print("- and more...")
    
    # Test device detection
    device = get_device()
    print(f"Detected device: {device}")
    
    # Test random seed setting
    set_random_seeds(42)
    print("Random seeds set for reproducibility")