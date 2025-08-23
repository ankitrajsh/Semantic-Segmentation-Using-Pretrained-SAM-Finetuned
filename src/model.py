   """
Model definitions for prostate cancer semantic segmentation.

This module contains model architectures and utilities for training
SAM-based segmentation models on prostate histopathology data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
from typing import Dict, List, Optional, Tuple
import numpy as np


class SAMProstateSegmentationModel(nn.Module):
    """
    SAM model adapted for prostate cancer segmentation.
    
    This class wraps the SAM model and adds prostate-specific
    modifications for better segmentation performance.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/sam-vit-base",
        num_classes: int = 6,
        freeze_encoder: bool = False,
        freeze_prompt_encoder: bool = True
    ):
        super().__init__()
        
        # Load pretrained SAM model
        self.sam = SamModel.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Freeze components if specified
        if freeze_encoder:
            self._freeze_vision_encoder()
        
        if freeze_prompt_encoder:
            self._freeze_prompt_encoder()
    
    def _freeze_vision_encoder(self):
        """Freeze the vision encoder parameters."""
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
    
    def _freeze_prompt_encoder(self):
        """Freeze the prompt encoder parameters."""
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
        input_points: Optional[torch.Tensor] = None,
        input_labels: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        def forward(
        self,
        pixel_values: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-class SAM model.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            input_boxes: Bounding box prompts [B, N, 4]
            multimask_output: Whether to output multiple masks
            
        Returns:
            Dictionary containing:
                - pred_masks: Class logits [B, num_classes, H, W]
                - iou_predictions: IoU predictions for each class
        """
        # Get SAM outputs
        outputs = self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )
        
        return outputs
    
    def predict_classes(self, pixel_values: torch.Tensor, input_boxes: Optional[torch.Tensor] = None):
        """
        Predict class probabilities and segmentation maps.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            input_boxes: Bounding box prompts [B, N, 4]
            
        Returns:
            Dictionary containing:
                - logits: Raw class logits [B, num_classes, H, W]
                - probabilities: Class probabilities [B, num_classes, H, W]
                - predicted_classes: Final segmentation [B, H, W]
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_boxes)
            
            # Get class logits
            logits = outputs.pred_masks  # [B, num_classes, H, W]
            
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted class for each pixel
            predicted_classes = torch.argmax(probabilities, dim=1)  # [B, H, W]
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predicted_classes': predicted_classes,
                'class_names': self.class_names
            }
    
    def get_class_weights(self, class_distribution: Optional[List[float]] = None) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            class_distribution: Optional list of class frequencies
            
        Returns:
            Tensor of class weights
        """
        if class_distribution is None:
            # Default weights based on typical medical imaging imbalance
            # Background gets lower weight, rare classes get higher weight
            weights = [0.1, 1.0, 2.0, 2.5, 3.0, 1.5]  # Background, Benign, G3, G4, G5, Stroma
        else:
            # Calculate inverse frequency weights
            total = sum(class_distribution)
            weights = [total / (len(class_distribution) * freq) for freq in class_distribution]
        
        return torch.tensor(weights, dtype=torch.float32)


class MultiClassLoss(nn.Module):
    """
    Combined loss function for multi-class segmentation.
    
    Combines Cross-Entropy loss with Dice loss for better performance
    on medical segmentation tasks.
    """
    
    def __init__(self, num_classes: int = 6, class_weights: Optional[torch.Tensor] = None,
                 ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions [B, num_classes, H, W]
            targets: Ground truth class indices [B, H, W]
            
        Returns:
            Combined loss value
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Dice loss for each class
        dice_loss = self._dice_loss(predictions, targets)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss
    
    def _dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Calculate multi-class Dice loss.
        
        Args:
            predictions: Model predictions [B, num_classes, H, W]
            targets: Ground truth class indices [B, H, W]
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        # Convert predictions to probabilities
        pred_probs = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(self.num_classes):
            pred_class = pred_probs[:, i, :, :]
            target_class = targets_one_hot[:, i, :, :]
            
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        # Average Dice score across classes
        mean_dice = torch.stack(dice_scores).mean()
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - mean_dice


def create_multiclass_sam_model(
    model_name: str = "facebook/sam-vit-base",
    num_classes: int = 6,
    freeze_encoder: bool = True,
    class_weights: Optional[List[float]] = None
) -> Tuple[MultiClassSAMModel, MultiClassLoss]:
    """
    Factory function to create a multi-class SAM model and loss function.
    
    Args:
        model_name: HuggingFace model name
        num_classes: Number of segmentation classes
        freeze_encoder: Whether to freeze encoder components
        class_weights: Optional class weights for loss function
        
    Returns:
        Tuple of (model, loss_function)
    """
    # Create model
    model = MultiClassSAMModel(
        model_name=model_name,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder
    )
    
    # Create loss function
    weights_tensor = None
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    loss_fn = MultiClassLoss(
        num_classes=num_classes,
        class_weights=weights_tensor
    )
    
    return model, loss_fn
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Print trainable components
        print("\nTrainable components:")
        for name, module in self.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_params > 0:
                print(f"  {name}: {module_params:,} parameters")


class MultiClassSAMModel(nn.Module):
    """
    Multi-class segmentation model based on SAM.
    
    This model extends SAM to handle multiple classes for
    prostate cancer Gleason grading with proper class values (0-5).
    
    Classes:
        0: Background
        1: Benign  
        2: Gleason Pattern 3
        3: Gleason Pattern 4
        4: Gleason Pattern 5
        5: Stroma
    """
    
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        num_classes: int = 6,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        self.sam = SamModel.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Freeze encoder components if specified
        if freeze_encoder:
            self._freeze_encoders()
        
        # Modify mask decoder for multi-class output
        self._modify_mask_decoder()
        
        # Class information
        self.class_names = {
            0: "Background",
            1: "Benign", 
            2: "Gleason_3",
            3: "Gleason_4", 
            4: "Gleason_5",
            5: "Stroma"
        }
    
    def _freeze_encoders(self):
        """Freeze vision encoder and prompt encoder."""
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
        
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
    
    def _modify_mask_decoder(self):
        """
        Modify mask decoder to output multiple classes instead of binary masks.
        
        This changes the final layer to output num_classes channels where each
        channel represents the logits for a specific class.
        """
        # Access the mask decoder's output upscaling layers
        upscaling_layers = self.sam.mask_decoder.output_upscaling
        
        # Get the original output dimension from the last conv layer
        if hasattr(upscaling_layers[-1], 'out_channels'):
            original_output_dim = upscaling_layers[-1].out_channels
        else:
            # Fallback: check the second-to-last layer
            original_output_dim = upscaling_layers[-2].out_channels
        
        # Replace the final layer to output num_classes channels
        final_layer = nn.Conv2d(
            in_channels=original_output_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            bias=True
        )
        
        # Initialize the new layer
        nn.init.kaiming_normal_(final_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(final_layer.bias)
        
        # Replace the layer
        upscaling_layers[-1] = final_layer
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-class output."""
        
        # Get SAM outputs
        sam_outputs = self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )
        
        # Get binary masks from SAM
        binary_masks = sam_outputs.pred_masks  # [B, 1, H, W]
        
        # Apply classifier to get multi-class predictions
        class_logits = self.classifier(binary_masks)  # [B, num_classes, H, W]
        
        return {
            'pred_masks': class_logits,
            'binary_masks': binary_masks,
            'iou_predictions': sam_outputs.iou_predictions
        }


class EnsembleSAMModel(nn.Module):
    """
    Ensemble of SAM models for improved performance.
    
    This model combines predictions from multiple SAM models
    trained with different configurations.
    """
    
    def __init__(self, model_configs: List[Dict]):
        super().__init__()
        
        self.models = nn.ModuleList()
        for config in model_configs:
            model = SAMProstateSegmentationModel(**config)
            self.models.append(model)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        
        all_predictions = []
        all_iou_predictions = []
        
        for model in self.models:
            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=multimask_output
            )
            all_predictions.append(outputs.pred_masks)
            all_iou_predictions.append(outputs.iou_predictions)
        
        # Average predictions
        ensemble_pred = torch.stack(all_predictions).mean(dim=0)
        ensemble_iou = torch.stack(all_iou_predictions).mean(dim=0)
        
        return {
            'pred_masks': ensemble_pred,
            'iou_predictions': ensemble_iou
        }


def create_sam_model(
    model_name: str = "facebook/sam-vit-base",
    num_classes: int = 6,
    freeze_encoder: bool = False,
    device: str = "cuda"
) -> SAMProstateSegmentationModel:
    """
    Create and initialize a SAM model for prostate segmentation.
    
    Args:
        model_name: Name of the pretrained SAM model
        num_classes: Number of output classes
        freeze_encoder: Whether to freeze the vision encoder
        device: Device to move the model to
        
    Returns:
        Initialized SAM model
    """
    model = SAMProstateSegmentationModel(
        model_name=model_name,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder
    )
    
    model = model.to(device)
    
    # Print model information
    print(f"Created SAM model: {model_name}")
    model.print_trainable_parameters()
    
    return model


def load_pretrained_sam_model(
    checkpoint_path: str,
    model_name: str = "facebook/sam-vit-base",
    num_classes: int = 6,
    device: str = "cuda"
) -> SAMProstateSegmentationModel:
    """
    Load a pretrained SAM model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_name: Name of the base SAM model
        num_classes: Number of output classes
        device: Device to load the model on
        
    Returns:
        Loaded SAM model
    """
    model = SAMProstateSegmentationModel(
        model_name=model_name,
        num_classes=num_classes
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded SAM model from: {checkpoint_path}")
    
    return model


class SAMModelWithMetrics(nn.Module):
    """
    SAM model wrapper that computes metrics during forward pass.
    """
    
    def __init__(self, base_model: SAMProstateSegmentationModel):
        super().__init__()
        self.base_model = base_model
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
        ground_truth_masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with metric computation."""
        
        # Get base model outputs
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )
        
        # Compute metrics if ground truth is provided
        if ground_truth_masks is not None:
            pred_masks = torch.sigmoid(outputs.pred_masks)
            
            # Dice coefficient
            dice = self._compute_dice(pred_masks, ground_truth_masks)
            outputs['dice_score'] = dice
            
            # IoU
            iou = self._compute_iou(pred_masks, ground_truth_masks)
            outputs['iou_score'] = iou
            
            # Pixel accuracy
            accuracy = self._compute_accuracy(pred_masks, ground_truth_masks)
            outputs['pixel_accuracy'] = accuracy
        
        return outputs
    
    def _compute_dice(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute Dice coefficient."""
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def _compute_iou(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute IoU score."""
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)
    
    def _compute_accuracy(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute pixel accuracy."""
        pred_binary = (pred > 0.5).float()
        correct = (pred_binary == target).sum()
        total = target.numel()
        return correct / total


def model_summary(model: nn.Module, input_size: Tuple[int, ...]):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (excluding batch dimension)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        print("torchsummary not installed. Install with: pip install torchsummary")
        
        # Basic summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = create_sam_model(device=device)
    
    # Print model summary
    print("\nModel Architecture Summary:")
    model_summary(model, (3, 256, 256))