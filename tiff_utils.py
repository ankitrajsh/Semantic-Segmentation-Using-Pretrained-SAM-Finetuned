"""
TIFF Image Utilities for Medical Imaging
========================================

This module provides utilities for handling TIFF files commonly used in 
medical imaging, particularly histopathology and microscopy images.

TIFF files can have various formats:
- Grayscale or RGB
- 8-bit, 16-bit, or 32-bit
- Single or multi-page
- Channel-first or channel-last
"""

import numpy as np
import tifffile as tiff
from PIL import Image
import os
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class TIFFHandler:
    """Handler for TIFF files in medical imaging contexts."""
    
    def __init__(self):
        self.supported_formats = ['.tiff', '.tif', '.TIF', '.TIFF']
    
    def load_image(self, file_path: str, target_channels: int = 3) -> np.ndarray:
        """
        Load TIFF image and normalize to standard format.
        
        Args:
            file_path: Path to TIFF file
            target_channels: Target number of channels (1 for grayscale, 3 for RGB)
            
        Returns:
            Image array with shape (H, W, C) and dtype uint8
        """
        if not self.is_tiff_file(file_path):
            raise ValueError(f"File {file_path} is not a TIFF file")
        
        # Load TIFF file
        image_array = tiff.imread(file_path)
        
        logger.info(f"Loaded TIFF: {file_path}")
        logger.info(f"Original shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        # Handle different TIFF formats
        image_array = self._normalize_tiff_format(image_array, target_channels)
        
        logger.info(f"Normalized shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        return image_array
    
    def load_mask(self, file_path: str, expected_classes: Optional[int] = None) -> np.ndarray:
        """
        Load TIFF mask and validate class values.
        
        Args:
            file_path: Path to TIFF mask file
            expected_classes: Expected number of classes (for validation)
            
        Returns:
            Mask array with shape (H, W) containing class indices
        """
        if not self.is_tiff_file(file_path):
            raise ValueError(f"File {file_path} is not a TIFF file")
        
        # Load TIFF mask
        mask_array = tiff.imread(file_path)
        
        logger.info(f"Loaded TIFF mask: {file_path}")
        logger.info(f"Original shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        
        # Convert to 2D if needed
        if len(mask_array.shape) == 3:
            if mask_array.shape[-1] == 3:
                # RGB mask - convert to class indices
                mask_array = self._rgb_to_class_indices(mask_array)
            else:
                # Multi-channel - take first channel
                mask_array = mask_array[:, :, 0]
        
        # Ensure integer type
        mask_array = mask_array.astype(np.uint8)
        
        # Validate class values
        unique_values = np.unique(mask_array)
        logger.info(f"Mask contains classes: {unique_values}")
        
        if expected_classes is not None:
            max_class = expected_classes - 1
            invalid_values = unique_values[unique_values > max_class]
            if len(invalid_values) > 0:
                logger.warning(f"Found invalid class values: {invalid_values} (max allowed: {max_class})")
                # Clip invalid values
                mask_array = np.clip(mask_array, 0, max_class)
        
        return mask_array
    
    def save_tiff(self, array: np.ndarray, file_path: str, compress: bool = True):
        """
        Save array as TIFF file.
        
        Args:
            array: Image or mask array to save
            file_path: Output file path
            compress: Whether to apply compression
        """
        # Determine compression
        compression = 'lzw' if compress else None
        
        # Save TIFF
        tiff.imwrite(file_path, array, compression=compression)
        
        logger.info(f"Saved TIFF: {file_path} (shape: {array.shape}, dtype: {array.dtype})")
    
    def is_tiff_file(self, file_path: str) -> bool:
        """Check if file is a TIFF file."""
        return any(file_path.endswith(ext) for ext in self.supported_formats)
    
    def _normalize_tiff_format(self, image_array: np.ndarray, target_channels: int) -> np.ndarray:
        """
        Normalize TIFF image to standard format.
        
        Args:
            image_array: Raw TIFF array
            target_channels: Target number of channels
            
        Returns:
            Normalized array with shape (H, W, C)
        """
        # Handle different array dimensions
        if len(image_array.shape) == 2:
            # Grayscale image
            if target_channels == 3:
                # Convert to RGB by repeating channels
                image_array = np.stack([image_array] * 3, axis=-1)
            else:
                # Keep as grayscale but add channel dimension
                image_array = np.expand_dims(image_array, axis=-1)
        
        elif len(image_array.shape) == 3:
            # Multi-channel image
            if image_array.shape[0] < image_array.shape[-1]:
                # Likely channel-first (C, H, W) - convert to channel-last (H, W, C)
                image_array = np.transpose(image_array, (1, 2, 0))
            
            # Handle channel count
            if image_array.shape[-1] > target_channels:
                # Too many channels - take first target_channels
                image_array = image_array[:, :, :target_channels]
            elif image_array.shape[-1] < target_channels and target_channels == 3:
                # Too few channels - repeat to make RGB
                if image_array.shape[-1] == 1:
                    image_array = np.repeat(image_array, 3, axis=-1)
        
        # Normalize data type and range
        image_array = self._normalize_dtype(image_array)
        
        return image_array
    
    def _normalize_dtype(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize array data type to uint8 with range 0-255.
        
        Args:
            array: Input array
            
        Returns:
            Array with dtype uint8 and range 0-255
        """
        if array.dtype == np.uint8:
            return array
        
        # Handle different input types
        if array.dtype in [np.uint16, np.uint32]:
            # Scale down from higher bit depth
            max_val = np.iinfo(array.dtype).max
            array = (array.astype(np.float32) / max_val * 255).astype(np.uint8)
        
        elif array.dtype in [np.float32, np.float64]:
            # Handle floating point
            if array.max() <= 1.0:
                # Assume normalized to [0, 1]
                array = (array * 255).astype(np.uint8)
            else:
                # Assume in range [0, 255] or higher
                array = np.clip(array, 0, 255).astype(np.uint8)
        
        else:
            # Generic case - clip and convert
            array = np.clip(array, 0, 255).astype(np.uint8)
        
        return array
    
    def _rgb_to_class_indices(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class indices.
        
        Args:
            rgb_array: RGB mask array with shape (H, W, 3)
            
        Returns:
            Class index array with shape (H, W)
        """
        # Define standard color mapping for prostate cancer segmentation
        color_to_class = {
            (0, 0, 0): 0,        # Background - Black
            (0, 0, 255): 1,      # Benign - Blue
            (0, 255, 0): 2,      # Gleason 3 - Green
            (255, 255, 0): 3,    # Gleason 4 - Yellow
            (255, 0, 0): 4,      # Gleason 5 - Red
            (128, 0, 128): 5,    # Stroma - Purple
        }
        
        # Initialize class mask
        class_mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
        
        # Map colors to classes
        for color, class_id in color_to_class.items():
            # Create mask for this color
            color_mask = np.all(rgb_array == color, axis=2)
            class_mask[color_mask] = class_id
        
        return class_mask
    
    def analyze_tiff_file(self, file_path: str) -> dict:
        """
        Analyze TIFF file and return metadata.
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            Dictionary with file metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load TIFF file
        array = tiff.imread(file_path)
        
        # Basic metadata
        metadata = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'shape': array.shape,
            'dtype': str(array.dtype),
            'min_value': array.min(),
            'max_value': array.max(),
            'mean_value': array.mean(),
        }
        
        # Additional analysis for masks
        if len(array.shape) == 2:
            unique_values = np.unique(array)
            metadata['unique_values'] = unique_values.tolist()
            metadata['num_classes'] = len(unique_values)
            
            # Value distribution
            value_counts = {int(val): int(np.sum(array == val)) for val in unique_values}
            metadata['value_distribution'] = value_counts
        
        return metadata


def visualize_tiff_comparison(image_path: str, mask_path: str, 
                            class_names: Optional[dict] = None,
                            figsize: Tuple[int, int] = (15, 5)):
    """
    Visualize TIFF image and corresponding mask.
    
    Args:
        image_path: Path to TIFF image
        mask_path: Path to TIFF mask
        class_names: Dictionary mapping class IDs to names
        figsize: Figure size
    """
    handler = TIFFHandler()
    
    # Load image and mask
    image = handler.load_image(image_path, target_channels=3)
    mask = handler.load_mask(mask_path, expected_classes=6)
    
    # Default class names
    if class_names is None:
        class_names = {
            0: "Background", 1: "Benign", 2: "Gleason_3",
            3: "Gleason_4", 4: "Gleason_5", 5: "Stroma"
        }
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original TIFF Image")
    axes[0].axis('off')
    
    # Mask as grayscale
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=5)
    axes[1].set_title("Mask (Class Values)")
    axes[1].axis('off')
    
    # Color-coded mask
    colors = np.array([
        [0, 0, 0],        # Background - Black
        [0, 0, 255],      # Benign - Blue
        [0, 255, 0],      # Gleason 3 - Green
        [255, 255, 0],    # Gleason 4 - Yellow
        [255, 0, 0],      # Gleason 5 - Red
        [128, 0, 128],    # Stroma - Purple
    ]) / 255.0
    
    colored_mask = colors[mask]
    axes[2].imshow(colored_mask)
    axes[2].set_title("Color-Coded Mask")
    axes[2].axis('off')
    
    # Add class information
    unique_classes = np.unique(mask)
    class_info = [f"Class {c}: {class_names.get(c, 'Unknown')}" for c in unique_classes]
    plt.figtext(0.1, 0.02, "Classes present: " + ", ".join(class_info), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\\nImage shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Classes present: {unique_classes}")
    
    for class_id in unique_classes:
        count = np.sum(mask == class_id)
        percentage = (count / mask.size) * 100
        print(f"  {class_names.get(class_id, f'Class_{class_id}')}: {count} pixels ({percentage:.2f}%)")


def batch_analyze_tiff_directory(directory: str, file_pattern: str = "*.tif*") -> dict:
    """
    Analyze all TIFF files in a directory.
    
    Args:
        directory: Directory containing TIFF files
        file_pattern: Pattern to match files
        
    Returns:
        Dictionary with analysis results
    """
    import glob
    
    handler = TIFFHandler()
    
    # Find TIFF files
    pattern = os.path.join(directory, file_pattern)
    tiff_files = glob.glob(pattern)
    
    if not tiff_files:
        print(f"No TIFF files found in {directory} matching pattern {file_pattern}")
        return {}
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Analyze each file
    results = {}
    for file_path in tiff_files:
        try:
            metadata = handler.analyze_tiff_file(file_path)
            results[os.path.basename(file_path)] = metadata
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Summary statistics
    if results:
        shapes = [result['shape'] for result in results.values()]
        dtypes = [result['dtype'] for result in results.values()]
        
        print(f"\\nSummary:")
        print(f"  Total files analyzed: {len(results)}")
        print(f"  Common shapes: {set(shapes)}")
        print(f"  Data types: {set(dtypes)}")
        
        # For masks, analyze class distributions
        mask_files = [f for f, r in results.items() if 'unique_values' in r]
        if mask_files:
            print(f"  Mask files: {len(mask_files)}")
            all_classes = set()
            for f in mask_files:
                all_classes.update(results[f]['unique_values'])
            print(f"  All classes found: {sorted(all_classes)}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("TIFF Handler for Medical Imaging")
    print("================================")
    
    # Check if example files exist
    example_image = "data/images/example.tiff"
    example_mask = "data/masks/example.tiff"
    
    if os.path.exists(example_image) and os.path.exists(example_mask):
        print("\\nVisualizing example TIFF files...")
        visualize_tiff_comparison(example_image, example_mask)
    else:
        print(f"\\nTo test TIFF visualization, place files at:")
        print(f"  Image: {example_image}")
        print(f"  Mask: {example_mask}")
    
    # Analyze directory if it exists
    if os.path.exists("data/images"):
        print("\\nAnalyzing TIFF files in data/images/...")
        results = batch_analyze_tiff_directory("data/images")
    
    if os.path.exists("data/masks"):
        print("\\nAnalyzing TIFF files in data/masks/...")
        results = batch_analyze_tiff_directory("data/masks")
