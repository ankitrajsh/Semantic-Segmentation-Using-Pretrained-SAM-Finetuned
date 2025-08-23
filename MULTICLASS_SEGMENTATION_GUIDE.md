# Multi-Class Semantic Segmentation Pipeline for Prostate Cancer

[![Medical AI](https://img.shields.io/badge/Medical-AI-brightgreen)](https://github.com/ankitrajsh/Semantic-Segmentation-Using-Pretrained-SAM-Finetuned)
[![SAM](https://img.shields.io/badge/Model-SAM--ViT--Base16-blue)](https://segment-anything.com/)
[![TIFF Support](https://img.shields.io/badge/TIFF-Supported-orange)](https://en.wikipedia.org/wiki/TIFF)
[![Multi-Class](https://img.shields.io/badge/Segmentation-Multi--Class-red)](https://en.wikipedia.org/wiki/Semantic_segmentation)

## 🎯 Quick Overview

This implementation provides **true semantic segmentation** for prostate cancer histopathology using a modified Segment Anything Model (SAM). Unlike binary segmentation, each pixel receives a specific class label (0-5) for comprehensive tissue classification and Gleason grading.

### 🏥 Clinical Classes
- **Class 0**: Background (Black)
- **Class 1**: Benign tissue (Blue) 
- **Class 2**: Gleason Pattern 3 (Green) - Well-differentiated
- **Class 3**: Gleason Pattern 4 (Yellow) - Moderately differentiated  
- **Class 4**: Gleason Pattern 5 (Red) - Poorly differentiated
- **Class 5**: Stroma (Purple) - Supporting tissue

### 🚀 Quick Start
```bash
# Train with TIFF support
python train_multiclass_sam.py

# Convert RGB masks to class indices  
python convert_dataset.py --image_dir data/images --mask_dir data/masks --convert_masks
```

---

## Overview

This document explains how your SAM-based model performs **true semantic segmentation** with distinct class values (0-5) for each pixel, rather than binary segmentation.

## 🎯 Multi-Class Segmentation Architecture

### 1. Input-Output Pipeline

```
Input Image (256×256×3)
         ↓
┌─────────────────────┐
│  SAM Vision Encoder │ ← ViT-Base16 (Frozen)
│  Feature Extraction │   
└─────────────────────┘
         ↓
┌─────────────────────┐
│  SAM Prompt Encoder │ ← Bounding Box Processing (Frozen)
│  Box → Embeddings   │   
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Modified Mask      │ ← **KEY CHANGE: 1→6 output channels**
│  Decoder            │   
└─────────────────────┘
         ↓
Class Logits (256×256×6)
         ↓
Softmax → Probabilities (256×256×6)
         ↓
ArgMax → Final Segmentation (256×256) with values 0-5
```

### 2. Key Architectural Changes

| Component | Original SAM | **Your Multi-Class SAM** |
|-----------|--------------|---------------------------|
| **Final Layer** | Conv2d(?, 1) | **Conv2d(?, 6)** |
| **Output Shape** | [B, 1, H, W] | **[B, 6, H, W]** |
| **Loss Function** | Binary Cross-Entropy | **Multi-class Cross-Entropy** |
| **Prediction** | Sigmoid → Binary | **Softmax → ArgMax → Classes** |

## 🔧 Implementation Details

### 1. Modified Mask Decoder
```python
class SimplifiedMultiClassSAM(nn.Module):
    def _modify_mask_decoder(self):
        """Change final layer from 1 to 6 output channels"""
        upscaling = self.sam.mask_decoder.output_upscaling
        
        # Find and replace final conv layer
        for i, layer in enumerate(upscaling):
            if isinstance(layer, nn.Conv2d) and i == len(upscaling) - 1:
                new_layer = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=6,  # 6 classes instead of 1
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                )
                upscaling[i] = new_layer
```

### 2. Multi-Class Dataset Format
```python
# Your masks should contain class values, not RGB colors
mask_example = np.array([
    [0, 0, 1, 1, 2],  # Background → Benign → Gleason 3
    [1, 2, 3, 4, 5],  # All classes present
    [3, 3, 4, 4, 2]   # Mixed cancer patterns
])

# NOT RGB images like [255, 0, 0] for red
# BUT integer class indices: 0, 1, 2, 3, 4, 5
```

### 3. Loss Function & Training
```python
# Multi-class Cross-Entropy with class weights
class_weights = torch.tensor([0.1, 1.0, 2.0, 2.5, 3.0, 1.5])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Forward pass
outputs = model(pixel_values, input_boxes)
logits = outputs.pred_masks  # [B, 6, H, W]
loss = criterion(logits, ground_truth)  # ground_truth: [B, H, W] with values 0-5

# Prediction
probabilities = F.softmax(logits, dim=1)  # [B, 6, H, W]
predicted_classes = torch.argmax(probabilities, dim=1)  # [B, H, W]
```

## 📊 Clinical Class Definitions

### Class Mapping
```python
CLASSES = {
    0: "Background",     # Non-tissue areas
    1: "Benign",         # Normal prostate glands
    2: "Gleason_3",      # Well-differentiated adenocarcinoma
    3: "Gleason_4",      # Moderately differentiated adenocarcinoma  
    4: "Gleason_5",      # Poorly differentiated adenocarcinoma
    5: "Stroma"          # Supporting connective tissue
}
```

### Clinical Significance
- **Gleason 3**: Lower grade cancer, better prognosis
- **Gleason 4**: Intermediate grade, moderate risk
- **Gleason 5**: High grade cancer, aggressive treatment needed
- **Gleason Score**: Primary + Secondary pattern (e.g., 3+4=7)

## 🔄 Complete Workflow

### 1. Data Preparation
```bash
# Your TIFF mask files should contain integer values 0-5
# Verify with:
import tifffile as tiff
import numpy as np

mask = tiff.imread("data/masks/sample_mask.tiff")
print("Unique values:", np.unique(mask))
# Should show: [0 1 2 3 4 5] or subset

# Use the conversion utility if you have RGB masks:
python convert_dataset.py --image_dir data/images --mask_dir data/masks --convert_masks
```

### 2. TIFF-Specific Handling
```python
from tiff_utils import TIFFHandler

# Analyze your TIFF files
handler = TIFFHandler()
metadata = handler.analyze_tiff_file("data/masks/example.tiff")
print("File info:", metadata)

# Load and validate TIFF mask
mask = handler.load_mask("data/masks/example.tiff", expected_classes=6)
print("Mask shape:", mask.shape)
print("Classes found:", np.unique(mask))
```

### 2. Training Pipeline
```python
# 1. Load multi-class dataset
dataset = MultiClassSAMDataset(image_dir, mask_dir, processor)

# 2. Initialize model with 6 classes
model = SimplifiedMultiClassSAM(num_classes=6)

# 3. Use proper loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 4. Train with class-aware metrics
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(pixel_values, input_boxes)
        loss = criterion(outputs.pred_masks, ground_truth)
        # ... training loop
```

### 3. Evaluation Metrics
```python
# Per-class metrics
for class_id in range(6):
    pred_binary = (predictions == class_id)
    gt_binary = (ground_truth == class_id)
    
    dice = dice_coefficient(pred_binary, gt_binary)
    iou = iou_score(pred_binary, gt_binary)
    
    print(f"Class {class_id} - Dice: {dice:.3f}, IoU: {iou:.3f}")

# Overall metrics
overall_accuracy = (predictions == ground_truth).mean()
mean_dice = calculate_mean_dice(predictions, ground_truth)
```

## 🎨 Visualization

### Color-Coded Results
```python
# Define colors for each class
colors = np.array([
    [0, 0, 0],        # Background - Black
    [0, 0, 255],      # Benign - Blue
    [0, 255, 0],      # Gleason 3 - Green
    [255, 255, 0],    # Gleason 4 - Yellow
    [255, 0, 0],      # Gleason 5 - Red
    [128, 0, 128],    # Stroma - Purple
])

# Convert class indices to RGB for visualization
rgb_mask = colors[predicted_classes]
plt.imshow(rgb_mask)
```

## 🚀 Usage Examples

### Simple Training
```bash
# Basic multi-class training
python train_multiclass_sam.py
```

### Advanced Training
```bash
# Full-featured implementation
python semantic_segmentation_multiclass.py
```

### Custom Dataset
```python
# Load your data
dataset = MultiClassSAMDataset("path/to/images", "path/to/masks", processor)

# Verify class distribution
class_counts = {}
for i in range(len(dataset)):
    mask = dataset[i]["ground_truth_mask"].numpy()
    unique, counts = np.unique(mask, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_counts[class_id] = class_counts.get(class_id, 0) + count

print("Class distribution:", class_counts)
```

## ⚠️ Important Notes

1. **Mask Format**: Your masks MUST contain integer values 0-5, not RGB colors
2. **Loss Function**: Use CrossEntropyLoss, not binary losses  
3. **Evaluation**: Calculate per-class metrics for clinical relevance
4. **Class Imbalance**: Use class weights to handle medical data imbalance
5. **Validation**: Always verify mask class values before training

## 🔧 Troubleshooting

### Common Issues
```python
# Issue: RGB masks instead of class indices
# Solution: Convert RGB to class indices
def rgb_to_class(rgb_mask):
    color_to_class = {
        (0, 0, 0): 0,      # Black → Background
        (0, 0, 255): 1,    # Blue → Benign
        (0, 255, 0): 2,    # Green → Gleason 3
        (255, 255, 0): 3,  # Yellow → Gleason 4
        (255, 0, 0): 4,    # Red → Gleason 5
        (128, 0, 128): 5,  # Purple → Stroma
    }
    class_mask = np.zeros(rgb_mask.shape[:2])
    for color, class_id in color_to_class.items():
        mask = np.all(rgb_mask == color, axis=2)
        class_mask[mask] = class_id
    return class_mask

# Issue: Wrong loss function
# Solution: Use CrossEntropyLoss for multi-class
criterion = nn.CrossEntropyLoss()  # NOT BCELoss or DiceLoss

# Issue: Wrong output interpretation
# Solution: Use argmax for final prediction
predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
```

This multi-class approach gives you comprehensive tissue classification essential for accurate prostate cancer diagnosis and Gleason grading!
