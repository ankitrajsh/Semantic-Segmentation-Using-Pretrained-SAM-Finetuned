# Prostate Cancer Semantic Segmentation Using Fine-tuned SAM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements semantic segmentation for prostate cancer histopathology images using a fine-tuned **Segment Anything Model (SAM)** with **Vision Transformer Base-16** backbone. The system performs automated Gleason grading classification, identifying different tissue patterns crucial for prostate cancer diagnosis and treatment planning.

### Key Features

- 🔬 **Medical Image Segmentation**: Specialized for prostate histopathology images
- 🧠 **SAM with ViT-Base16**: Leverages state-of-the-art vision transformer architecture
- 🎯 **Multi-class Gleason Grading**: Classifies Gleason patterns 3, 4, and 5
- 📊 **Comprehensive Evaluation**: Multiple metrics including Dice, IoU, and Cohen's Kappa
- ⚡ **GPU Accelerated**: Optimized for CUDA-enabled training and inference

## Dataset Classes & Multi-Class Segmentation

The model performs **true semantic segmentation** where each pixel is assigned a specific class value (0-5), enabling comprehensive tissue classification:

| Class | Value | Color | Description | Clinical Significance |
|-------|-------|--------|-------------|---------------------|
| 0 | Background | Black | Non-tissue regions | N/A |
| 1 | Benign | Blue | Normal prostate tissue | Healthy glandular structures |
| 2 | Gleason 3 | Green | Well-differentiated cancer | Lower grade, better prognosis |
| 3 | Gleason 4 | Yellow | Moderately differentiated cancer | Intermediate grade |
| 4 | Gleason 5 | Red | Poorly differentiated cancer | Higher grade, requires treatment |
| 5 | Stroma | Purple | Supporting connective tissue | Microenvironment context |

### Multi-Class Implementation

Unlike binary segmentation, this implementation uses:
- **Class-based masks**: Each pixel contains integer values 0-5
- **Cross-Entropy Loss**: Optimized for multi-class classification
- **Class weights**: Handles imbalanced medical data
- **Per-class metrics**: Individual evaluation for each tissue type

```python
# Example mask format (not binary!)
mask = np.array([
    [0, 0, 1, 1, 2],  # Background, Benign, Gleason 3
    [1, 2, 3, 4, 5],  # All tissue types
    [3, 3, 4, 4, 5]   # Cancer progression
])
```

## Project Structure

```
Semantic-Segmentation-Using-Pretrained-SAM-Finetuned/
├── README.md                           # Project documentation
├── workflow.md                         # Detailed workflow guide
├── requirements.txt                    # Python dependencies
├── Main_final_SAM.py                  # Main training pipeline
├── SAM_Model_segmentation.py          # Model implementation
├── train.py                           # Training utilities
├── Patches.py                         # Patch extraction
├── patch_for_mask.py                  # Mask processing
├── Resize.py                          # Image resizing
├── data/                              # Dataset directory
│   ├── images/                        # Training images
│   └── masks/                         # Corresponding masks
├── src/                               # Source code modules
│   ├── dataset.py                     # Dataset classes
│   ├── model.py                       # Model definitions
│   ├── train.py                       # Training functions
│   ├── evaluate.py                    # Evaluation metrics
│   └── utils.py                       # Utility functions
└── ViT_new/                          # ViT experiments
    └── Transferlearning_vit.py
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ankitrajsh/Semantic-Segmentation-Using-Pretrained-SAM-Finetuned.git
cd Semantic-Segmentation-Using-Pretrained-SAM-Finetuned
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets matplotlib transformers
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install monai albumentations tqdm scikit-learn
```

### 4. Prepare Dataset

Organize your prostate histopathology data (supports **TIFF, PNG, JPG formats**):

```bash
mkdir -p data/{images,masks}
# Place your image patches in data/images/ (.tiff, .tif, .png, .jpg)
# Place corresponding mask patches in data/masks/ (.tiff, .tif, .png, .jpg)
```

**TIFF Support**: This implementation fully supports medical imaging TIFF files, which are common in histopathology:
- Multi-channel TIFF images
- 16-bit and 32-bit TIFF files (auto-normalized to 8-bit)
- Both grayscale and RGB TIFF formats
- Compressed and uncompressed TIFF files

## Usage

### Multi-Class Semantic Segmentation

#### Quick Start - Multi-Class Training

**1. Basic Multi-Class Training (Recommended)**
```bash
# Train with the simplified multi-class approach
python train_multiclass_sam.py
```

**2. Advanced Multi-Class Training**
```bash
# Use the comprehensive implementation with advanced features
python semantic_segmentation_multiclass.py
```

**3. Using Your Own Data (TIFF Support)**
```python
from semantic_segmentation_multiclass import MultiClassSAMDataset, MultiClassSAMModel
from transformers import SamProcessor

# Initialize processor and dataset (supports .tiff, .tif, .png, .jpg)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
dataset = MultiClassSAMDataset(
    image_dir="data/images",      # TIFF files supported!
    mask_dir="data/masks",        # TIFF masks with class values 0-5
    processor=processor
)

# Create model for 6 classes
model = MultiClassSAMModel(num_classes=6)

# IMPORTANT: Your TIFF masks should contain integer values:
# 0 = Background, 1 = Benign, 2 = Gleason 3, 
# 3 = Gleason 4, 4 = Gleason 5, 5 = Stroma
```

**TIFF File Analysis**:
```python
# Analyze your TIFF files
from tiff_utils import TIFFHandler, visualize_tiff_comparison

# Check what's in your TIFF files
handler = TIFFHandler()
metadata = handler.analyze_tiff_file("data/masks/example.tiff")
print("Mask classes:", metadata['unique_values'])

# Visualize TIFF image and mask
visualize_tiff_comparison("data/images/img.tiff", "data/masks/mask.tiff")
```

#### Key Differences from Binary Segmentation

| Aspect | Binary SAM | **Multi-Class SAM** |
|--------|------------|---------------------|
| **Output** | Single mask (0/1) | **Class indices (0-5)** |
| **Loss** | Binary Cross-Entropy | **Multi-class Cross-Entropy** |
| **Metrics** | Binary Dice/IoU | **Per-class + Mean metrics** |
| **Final Layer** | 1 channel | **6 channels (num_classes)** |
| **Clinical Value** | Limited | **Full tissue classification** |

### Traditional Workflow (Legacy)

1. **Data Preprocessing** (if working with large images):
```bash
python Patches.py  # Extract patches from large images
python patch_for_mask.py  # Process corresponding masks
```

2. **Train the Model**:
```bash
python Main_final_SAM.py
```

3. **Evaluate Results**:
```bash
python src/evaluate.py
```

### Training Configuration

Key training parameters in `Main_final_SAM.py`:

```python
# Training settings
num_epochs = 4
batch_size = 2
learning_rate = 5e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
model_name = "facebook/sam-vit-base"
loss_function = DiceCELoss()  # Combines Dice and Cross-Entropy
```

### Custom Dataset

To use your own dataset, ensure the following structure:

```python
# Image and mask files should have matching names
data/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

## Model Architecture

The system uses **SAM (Segment Anything Model)** with the following modifications:

- **Backbone**: ViT-Base16 (Vision Transformer)
- **Fine-tuning**: Mask decoder adapted for medical segmentation
- **Input Size**: 256×256 patches
- **Output**: Multi-class segmentation masks

### Training Pipeline

1. **Patch Extraction**: Large histopathology images → 256×256 patches
2. **Data Loading**: Custom SAMDataset with bounding box prompts
3. **Model Training**: Fine-tune SAM decoder with DiceCE loss
4. **Validation**: Monitor Dice coefficient and IoU scores
5. **Inference**: Segment test images and reassemble patches

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **Dice Coefficient**: Measures overlap between predicted and ground truth
- **Intersection over Union (IoU)**: Measures segmentation accuracy
- **Cohen's Kappa**: Inter-rater agreement measure
- **Pixel Accuracy**: Overall correctness of pixel classification

## Results

Expected performance on prostate histopathology data:

| Metric | Target Performance |
|--------|-------------------|
| Mean Dice Coefficient | > 0.80 |
| Mean IoU | > 0.75 |
| Overall Accuracy | > 90% |
| Cohen's Kappa | > 0.75 |

## Advanced Features

### Transfer Learning Options

Explore different transfer learning approaches:

```bash
python transfer_learning_segmentation.py  # MaskFormer approach
python Transferlearning_new.py           # SegFormer approach
python TL_Imagenet1k_VIT.py             # Pure ViT approach
```

### Data Augmentation

The system includes robust data augmentation:

- Random cropping and resizing
- Horizontal/vertical flipping
- Color jittering
- Elastic transformations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Clinical Applications

This system can assist in:

- **Automated Gleason Grading**: Consistent cancer pattern identification
- **Pathologist Support**: Second opinion for complex cases
- **Research Studies**: Large-scale histopathology analysis
- **Treatment Planning**: Accurate tumor characterization

## Future Improvements

- [ ] Multi-scale feature extraction
- [ ] Attention mechanisms for better boundary detection
- [ ] Integration with genomic data
- [ ] Real-time inference optimization
- [ ] DICOM integration for clinical workflows

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{prostate_sam_segmentation,
  title={Prostate Cancer Semantic Segmentation Using Fine-tuned SAM},
  author={Ankit Sharma},
  year={2024},
  url={https://github.com/ankitrajsh/Semantic-Segmentation-Using-Pretrained-SAM-Finetuned}
}
```

## Acknowledgments

- Meta AI for the Segment Anything Model
- Hugging Face for the Transformers library
- MONAI for medical imaging utilities
- The medical imaging community for datasets and benchmarks

## Contact

For questions and support:
- GitHub Issues: [Create an issue](https://github.com/ankitrajsh/Semantic-Segmentation-Using-Pretrained-SAM-Finetuned/issues)
- Email: [Contact maintainer](mailto:ankit@example.com)

---

**⚠️ Medical Disclaimer**: This software is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.