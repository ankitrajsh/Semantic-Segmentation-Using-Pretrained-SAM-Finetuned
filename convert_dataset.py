"""
Dataset Conversion Utility for TIFF Medical Images
==================================================

This script helps convert and validate your TIFF dataset for multi-class
semantic segmentation with SAM.

Usage:
    python convert_dataset.py --image_dir /path/to/images --mask_dir /path/to/masks
"""

import os
import argparse
import numpy as np
from tiff_utils import TIFFHandler, batch_analyze_tiff_directory
import tifffile as tiff
from PIL import Image
import shutil
from tqdm import tqdm


def convert_rgb_masks_to_class_indices(mask_dir: str, output_dir: str):
    """
    Convert RGB TIFF masks to class index masks.
    
    Args:
        mask_dir: Directory containing RGB TIFF masks
        output_dir: Directory to save converted masks
    """
    handler = TIFFHandler()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIFF files
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.tiff', '.tif'))]
    
    print(f"Converting {len(mask_files)} mask files from RGB to class indices...")
    
    for mask_file in tqdm(mask_files, desc="Converting masks"):
        input_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)
        
        try:
            # Load RGB mask
            rgb_mask = tiff.imread(input_path)
            
            if len(rgb_mask.shape) == 3 and rgb_mask.shape[-1] == 3:
                # Convert RGB to class indices
                class_mask = handler._rgb_to_class_indices(rgb_mask)
                
                # Save as class indices
                handler.save_tiff(class_mask, output_path, compress=True)
                
                print(f"✓ Converted {mask_file}: {np.unique(rgb_mask.flatten())} → {np.unique(class_mask)}")
            else:
                print(f"⚠ Skipping {mask_file}: Not RGB format (shape: {rgb_mask.shape})")
                
        except Exception as e:
            print(f"✗ Error converting {mask_file}: {e}")


def validate_dataset(image_dir: str, mask_dir: str):
    """
    Validate that image and mask pairs are compatible.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
    """
    handler = TIFFHandler()
    
    # Get file lists
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) 
                        if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
    
    print(f"\\nDataset Validation")
    print(f"==================")
    print(f"Images: {len(image_files)} files")
    print(f"Masks: {len(mask_files)} files")
    
    # Check file count match
    if len(image_files) != len(mask_files):
        print(f"⚠ WARNING: Mismatch in file counts!")
        return False
    
    # Check file correspondence
    mismatched_files = []
    for img_file in image_files:
        # Try to find corresponding mask
        base_name = os.path.splitext(img_file)[0]
        mask_candidates = [f for f in mask_files if f.startswith(base_name)]
        
        if not mask_candidates:
            mismatched_files.append(img_file)
    
    if mismatched_files:
        print(f"⚠ WARNING: {len(mismatched_files)} images without corresponding masks:")
        for f in mismatched_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(mismatched_files) > 5:
            print(f"  ... and {len(mismatched_files) - 5} more")
        return False
    
    # Validate a few sample pairs
    sample_pairs = min(5, len(image_files))
    print(f"\\nValidating {sample_pairs} sample pairs...")
    
    for i in range(sample_pairs):
        img_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])
        
        try:
            # Load image and mask
            if handler.is_tiff_file(img_path):
                image = handler.load_image(img_path)
            else:
                image = np.array(Image.open(img_path))
            
            if handler.is_tiff_file(mask_path):
                mask = handler.load_mask(mask_path, expected_classes=6)
            else:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    mask = handler._rgb_to_class_indices(mask)
            
            # Check dimensions
            if image.shape[:2] != mask.shape[:2]:
                print(f"✗ {image_files[i]}: Size mismatch - Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
                return False
            
            # Check mask classes
            unique_classes = np.unique(mask)
            if np.max(unique_classes) > 5:
                print(f"✗ {mask_files[i]}: Invalid classes {unique_classes} (max should be 5)")
                return False
            
            print(f"✓ {image_files[i]}: Shape {image.shape[:2]}, Classes {unique_classes}")
            
        except Exception as e:
            print(f"✗ Error validating {image_files[i]}: {e}")
            return False
    
    print("\\n✓ Dataset validation passed!")
    return True


def analyze_class_distribution(mask_dir: str):
    """
    Analyze class distribution across all masks.
    
    Args:
        mask_dir: Directory containing masks
    """
    handler = TIFFHandler()
    
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))]
    
    print(f"\\nAnalyzing class distribution across {len(mask_files)} masks...")
    
    class_counts = {i: 0 for i in range(6)}
    total_pixels = 0
    
    for mask_file in tqdm(mask_files, desc="Analyzing masks"):
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
            if handler.is_tiff_file(mask_path):
                mask = handler.load_mask(mask_path, expected_classes=6)
            else:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    mask = handler._rgb_to_class_indices(mask)
            
            # Count pixels for each class
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < 6:
                    class_counts[class_id] += count
            
            total_pixels += mask.size
            
        except Exception as e:
            print(f"Error analyzing {mask_file}: {e}")
    
    # Print distribution
    class_names = {
        0: "Background", 1: "Benign", 2: "Gleason_3",
        3: "Gleason_4", 4: "Gleason_5", 5: "Stroma"
    }
    
    print(f"\\nClass Distribution:")
    print(f"==================")
    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"Class {class_id} ({class_names[class_id]}): {count:,} pixels ({percentage:.2f}%)")
    
    return class_counts


def setup_training_data(image_dir: str, mask_dir: str, output_dir: str, 
                       train_split: float = 0.8):
    """
    Setup training/validation split for the dataset.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        output_dir: Output directory for organized data
        train_split: Fraction of data for training
    """
    import random
    
    # Get file lists
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
    
    # Create output structure
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"\\nSetting up training data:")
    print(f"Training: {len(train_files)} samples")
    print(f"Validation: {len(val_files)} samples")
    
    # Copy files
    for split_name, file_list, img_out, mask_out in [
        ('Training', train_files, train_img_dir, train_mask_dir),
        ('Validation', val_files, val_img_dir, val_mask_dir)
    ]:
        print(f"\\nCopying {split_name} files...")
        for img_file in tqdm(file_list):
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(img_out, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Find and copy corresponding mask
            base_name = os.path.splitext(img_file)[0]
            mask_files = [f for f in os.listdir(mask_dir) if f.startswith(base_name)]
            
            if mask_files:
                mask_file = mask_files[0]  # Take first match
                src_mask = os.path.join(mask_dir, mask_file)
                dst_mask = os.path.join(mask_out, mask_file)
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"⚠ Warning: No mask found for {img_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert and validate TIFF dataset for SAM segmentation")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--mask_dir", required=True, help="Directory containing masks")
    parser.add_argument("--convert_masks", action="store_true", help="Convert RGB masks to class indices")
    parser.add_argument("--output_dir", help="Output directory for converted data")
    parser.add_argument("--setup_splits", action="store_true", help="Setup train/val splits")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    
    args = parser.parse_args()
    
    print("TIFF Dataset Conversion & Validation")
    print("====================================")
    
    # Check directories exist
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    if not os.path.exists(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        return
    
    # Analyze directories
    print(f"\\nAnalyzing image directory: {args.image_dir}")
    img_results = batch_analyze_tiff_directory(args.image_dir)
    
    print(f"\\nAnalyzing mask directory: {args.mask_dir}")
    mask_results = batch_analyze_tiff_directory(args.mask_dir)
    
    # Convert RGB masks if requested
    if args.convert_masks:
        if not args.output_dir:
            args.output_dir = args.mask_dir + "_converted"
        
        print(f"\\nConverting RGB masks to class indices...")
        convert_rgb_masks_to_class_indices(args.mask_dir, args.output_dir)
        args.mask_dir = args.output_dir  # Use converted masks for further analysis
    
    # Validate dataset
    validate_dataset(args.image_dir, args.mask_dir)
    
    # Analyze class distribution
    analyze_class_distribution(args.mask_dir)
    
    # Setup training splits if requested
    if args.setup_splits:
        if not args.output_dir:
            args.output_dir = "prepared_dataset"
        
        setup_training_data(args.image_dir, args.mask_dir, args.output_dir, args.train_split)
        
        print(f"\\nDataset prepared in: {args.output_dir}")
        print("Use the following paths for training:")
        print(f"  Train images: {os.path.join(args.output_dir, 'train', 'images')}")
        print(f"  Train masks: {os.path.join(args.output_dir, 'train', 'masks')}")
        print(f"  Val images: {os.path.join(args.output_dir, 'val', 'images')}")
        print(f"  Val masks: {os.path.join(args.output_dir, 'val', 'masks')}")


if __name__ == "__main__":
    main()
