# Prostate Cancer Semantic Segmentation Workflow

## Project Overview
This project implements semantic segmentation for prostate cancer histopathology images using the Segment Anything Model (SAM) with ViT-Base16 backbone for Gleason grading classification.

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Images    │───▶│   Preprocessing   │───▶│     Patches     │
│   & Masks       │    │   & Patching     │    │   (256x256)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluation    │◀───│   Fine-tuned     │◀───│  Data Loading   │
│   & Metrics     │    │   SAM Model      │    │   & Augment     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲
                                │
                       ┌─────────────────┐
                       │   Training      │
                       │   Loop          │
                       └─────────────────┘
```

## Workflow Steps

### 1. Data Preparation
- **Input**: Large histopathology images and corresponding masks
- **Process**: 
  - Convert images into 256x256 patches
  - Generate corresponding mask patches
  - Store patches with proper naming convention
- **Output**: Patch datasets for training

### 2. Data Loading & Preprocessing
- **Dataset Structure**:
  ```
  data/
  ├── images/          # Training image patches
  └── masks/           # Corresponding mask patches
  ```
- **Classes**: 
  - Background (0)
  - Benign (1) - Blue
  - Gleason Pattern 3 (2) - Green  
  - Gleason Pattern 4 (3) - Yellow
  - Gleason Pattern 5 (4) - Red
  - Stroma (5) - Purple

### 3. Model Architecture
- **Base Model**: SAM (Segment Anything Model)
- **Backbone**: ViT-Base16 (facebook/sam-vit-base)
- **Fine-tuning**: Mask decoder layers for prostate-specific segmentation
- **Loss Function**: DiceCE Loss (combination of Dice and Cross-Entropy)

### 4. Training Process
- **Epochs**: 4-10 (configurable)
- **Batch Size**: 2-4 (memory dependent)
- **Optimizer**: Adam
- **Learning Rate**: 5e-5
- **Metrics**: Dice Coefficient, IoU Score, Accuracy

### 5. Evaluation & Validation
- **Metrics**:
  - Dice Coefficient
  - Mean IoU
  - Cohen's Kappa
  - Pixel Accuracy
- **Validation**: Hold-out test set
- **Cross-validation**: For robust evaluation

### 6. Inference & Results
- **Input**: Test histopathology images
- **Process**: 
  - Patch extraction
  - Segmentation prediction
  - Patch reassembly
- **Output**: Segmented images with Gleason grade classification

## Key Features

### Multi-Class Segmentation
- Simultaneous detection of multiple tissue types
- Gleason pattern classification (3, 4, 5)
- Stroma and benign tissue identification

### Advanced Preprocessing
- Patch-based approach for handling large images
- Bounding box generation for SAM prompting
- Data augmentation for better generalization

### Robust Training
- DiceCE loss for handling class imbalance
- Multiple evaluation metrics
- GPU acceleration support

## Implementation Files

### Core Implementation
- `Main_final_SAM.py` - Complete training pipeline
- `SAM_Model_segmentation.py` - Model implementation
- `train.py` - Training utilities

### Data Processing
- `Patches.py` - Patch extraction
- `patch_for_mask.py` - Mask processing
- `Resize.py` - Image resizing utilities

### Evaluation
- `src/evaluate.py` - Evaluation metrics
- `transfer_learning_segmentation.py` - Transfer learning approach

### Utilities
- `src/dataset.py` - Dataset classes
- `src/model.py` - Model definitions
- `src/utils.py` - Utility functions

## Usage Instructions

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python Patches.py  # Extract patches from large images
python patch_for_mask.py  # Process corresponding masks
```

### 3. Training
```bash
python Main_final_SAM.py  # Full training pipeline
# or
python train.py  # Basic training
```

### 4. Evaluation
```bash
python src/evaluate.py  # Evaluate trained model
```

### 5. Inference
```bash
python SAM_Model_segmentation.py  # Run inference on test data
```

## Expected Outcomes

### Performance Metrics
- **Dice Coefficient**: >0.8 for major tissue types
- **Mean IoU**: >0.75 overall
- **Accuracy**: >90% for Gleason grading

### Clinical Applications
- Automated prostate cancer grading
- Reduced pathologist workload
- Consistent, reproducible results
- Support for treatment planning

## Future Improvements

### Model Enhancements
- Incorporate Vision Transformers (ViTs) for better global context
- Hybrid CNN-ViT architecture
- Multi-scale feature extraction

### Data & Training
- Larger, more diverse datasets
- Cross-dataset validation
- Advanced data augmentation (style transfer)
- Multi-center validation studies

### Clinical Integration
- Integration with genomic data
- Multi-class grading beyond Gleason 3-5
- Real-time inference capabilities
- DICOM integration for clinical workflows

## References
- SAM Paper: "Segment Anything" (Meta AI)
- Gleason Grading System for Prostate Cancer
- Vision Transformer Architecture
- Transfer Learning for Medical Imaging
