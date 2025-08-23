   """
Evaluation utilities for prostate cancer semantic segmentation.

This module provides comprehensive evaluation metrics and visualization
tools for assessing model performance on prostate histopathology data.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation.
    
    Includes metrics specific to medical image segmentation and
    multi-class evaluation for Gleason grading.
    """
    
    def __init__(self, num_classes: int = 6, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = [
                "Background", "Benign", "Gleason_3", 
                "Gleason_4", "Gleason_5", "Stroma"
            ]
        else:
            self.class_names = class_names
    
    def dice_coefficient(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute Dice coefficient (F1 score for segmentation).
        
        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W] or [B, C, H, W]
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient
        """
        if pred.dim() == 4:  # Multi-class
            dice_scores = []
            for c in range(pred.shape[1]):
                pred_c = pred[:, c]
                target_c = target[:, c] if target.dim() == 4 else (target == c).float()
                
                intersection = (pred_c * target_c).sum()
                dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
                dice_scores.append(dice)
            
            return torch.stack(dice_scores)
        else:  # Binary
            intersection = (pred * target).sum()
            return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def iou_score(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU) score.
        
        Args:
            pred: Predicted masks
            target: Ground truth masks
            smooth: Smoothing factor
            
        Returns:
            IoU score
        """
        if pred.dim() == 4:  # Multi-class
            iou_scores = []
            for c in range(pred.shape[1]):
                pred_c = (pred[:, c] > 0.5).float()
                target_c = target[:, c] if target.dim() == 4 else (target == c).float()
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum() - intersection
                iou = (intersection + smooth) / (union + smooth)
                iou_scores.append(iou)
            
            return torch.stack(iou_scores)
        else:  # Binary
            pred_binary = (pred > 0.5).float()
            intersection = (pred_binary * target).sum()
            union = pred_binary.sum() + target.sum() - intersection
            return (intersection + smooth) / (union + smooth)
    
    def pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise accuracy."""
        if pred.dim() == 4:  # Multi-class
            pred_classes = torch.argmax(pred, dim=1)
        else:
            pred_classes = (pred > 0.5).long()
        
        correct = (pred_classes == target).sum()
        total = target.numel()
        return correct.float() / total
    
    def mean_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean IoU across all classes."""
        iou_scores = self.iou_score(pred, target)
        return iou_scores.mean()
    
    def class_wise_metrics(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for each class.
        
        Returns:
            Dictionary with class-wise metrics
        """
        metrics = {}
        
        # Dice scores
        dice_scores = self.dice_coefficient(pred, target)
        for i, class_name in enumerate(self.class_names):
            if i < len(dice_scores):
                metrics[f"{class_name}_dice"] = dice_scores[i]
        
        # IoU scores
        iou_scores = self.iou_score(pred, target)
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_scores):
                metrics[f"{class_name}_iou"] = iou_scores[i]
        
        # Overall metrics
        metrics["mean_dice"] = dice_scores.mean()
        metrics["mean_iou"] = iou_scores.mean()
        metrics["pixel_accuracy"] = self.pixel_accuracy(pred, target)
        
        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation for prostate cancer segmentation.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str = "cuda",
        num_classes: int = 6
    ):
        self.model = model
        self.device = device
        self.metrics = SegmentationMetrics(num_classes)
        
        # Store results
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': [],
            'file_names': []
        }
    
    def evaluate_batch(
        self, 
        pixel_values: torch.Tensor,
        input_boxes: torch.Tensor,
        ground_truth_masks: torch.Tensor,
        file_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a single batch.
        
        Args:
            pixel_values: Input images
            input_boxes: Bounding box prompts
            ground_truth_masks: Ground truth segmentation masks
            file_names: Optional file names for tracking
            
        Returns:
            Dictionary with batch metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values.to(self.device),
                input_boxes=input_boxes.to(self.device),
                multimask_output=False
            )
            
            # Get predictions
            pred_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
            gt_masks = ground_truth_masks.float().to(self.device)
            
            # Compute metrics
            batch_metrics = self.metrics.class_wise_metrics(pred_masks, gt_masks)
            
            # Store results
            self.results['predictions'].extend(pred_masks.cpu().numpy())
            self.results['ground_truths'].extend(gt_masks.cpu().numpy())
            self.results['metrics'].append({k: v.item() for k, v in batch_metrics.items()})
            
            if file_names:
                self.results['file_names'].extend(file_names)
            
            return {k: v.item() for k, v in batch_metrics.items()}
    
    def evaluate_dataloader(
        self, 
        dataloader: torch.utils.data.DataLoader,
        save_results: bool = True,
        results_dir: str = "results"
    ) -> Dict[str, float]:
        """
        Evaluate model on entire dataloader.
        
        Args:
            dataloader: Data loader with test data
            save_results: Whether to save detailed results
            results_dir: Directory to save results
            
        Returns:
            Dictionary with overall metrics
        """
        from tqdm import tqdm
        
        print("Evaluating model...")
        
        all_metrics = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            file_names = batch.get('file_name', None)
            
            batch_metrics = self.evaluate_batch(
                pixel_values=batch['pixel_values'],
                input_boxes=batch['input_boxes'],
                ground_truth_masks=batch['ground_truth_mask'],
                file_names=file_names
            )
            
            all_metrics.append(batch_metrics)
        
        # Compute overall metrics
        overall_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            overall_metrics[key] = np.mean(values)
            overall_metrics[f"{key}_std"] = np.std(values)
        
        # Print results
        self._print_evaluation_results(overall_metrics)
        
        # Save results if requested
        if save_results:
            self._save_results(overall_metrics, results_dir)
        
        return overall_metrics
    
    def _print_evaluation_results(self, metrics: Dict[str, float]):
        """Print formatted evaluation results."""
        print("
" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Overall metrics
        print(f"Overall Metrics:")
        print(f"  Mean Dice Coefficient: {metrics['mean_dice']:.4f} ± {metrics['mean_dice_std']:.4f}")
        print(f"  Mean IoU:             {metrics['mean_iou']:.4f} ± {metrics['mean_iou_std']:.4f}")
        print(f"  Pixel Accuracy:       {metrics['pixel_accuracy']:.4f} ± {metrics['pixel_accuracy_std']:.4f}")
        
        # Class-wise metrics
        print(f"
Class-wise Metrics:")
        for class_name in self.metrics.class_names:
            dice_key = f"{class_name}_dice"
            iou_key = f"{class_name}_iou"
            
            if dice_key in metrics and iou_key in metrics:
                print(f"  {class_name:12s}: Dice={metrics[dice_key]:.4f}, IoU={metrics[iou_key]:.4f}")
        
        print("="*60)
    
    def _save_results(self, metrics: Dict[str, float], results_dir: str):
        """Save detailed evaluation results."""
        Path(results_dir).mkdir(exist_ok=True)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{results_dir}/evaluation_metrics.csv", index=False)
        
        # Save detailed results
        if self.results['file_names']:
            detailed_results = []
            for i, file_name in enumerate(self.results['file_names']):
                result_dict = {'file_name': file_name}
                result_dict.update(self.results['metrics'][i])
                detailed_results.append(result_dict)
            
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(f"{results_dir}/detailed_results.csv", index=False)
        
        print(f"Results saved to {results_dir}/")
    
    def visualize_predictions(
        self, 
        num_samples: int = 5,
        save_path: Optional[str] = None
    ):
        """
        Visualize model predictions vs ground truth.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        if not self.results['predictions']:
            print("No predictions available. Run evaluation first.")
            return
        
        num_samples = min(num_samples, len(self.results['predictions']))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            pred = self.results['predictions'][i]
            gt = self.results['ground_truths'][i]
            
            # Convert to binary for visualization
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # Original would need to be stored separately
            # For now, show ground truth as "original"
            axes[i, 0].imshow(gt, cmap='viridis')
            axes[i, 0].set_title(f'Ground Truth {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_binary, cmap='viridis')
            axes[i, 1].set_title(f'Prediction {i+1}')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(gt, alpha=0.7, cmap='Greens')
            axes[i, 2].imshow(pred_binary, alpha=0.5, cmap='Reds')
            axes[i, 2].set_title(f'Overlay {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix for multi-class segmentation.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.results['predictions']:
            print("No predictions available. Run evaluation first.")
            return None
        
        # Flatten all predictions and ground truths
        all_preds = []
        all_gts = []
        
        for pred, gt in zip(self.results['predictions'], self.results['ground_truths']):
            if pred.ndim == 3:  # Multi-class predictions
                pred_classes = np.argmax(pred, axis=0).flatten()
            else:  # Binary predictions
                pred_classes = (pred > 0.5).astype(int).flatten()
            
            gt_classes = gt.flatten().astype(int)
            
            all_preds.extend(pred_classes)
            all_gts.extend(gt_classes)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_gts, all_preds, labels=range(self.metrics.num_classes))
        
        return cm
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix heatmap."""
        cm = self.compute_confusion_matrix()
        
        if cm is None:
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.metrics.class_names,
            yticklabels=self.metrics.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compute_cohen_kappa(self) -> float:
        """Compute Cohen's Kappa score for inter-rater agreement."""
        if not self.results['predictions']:
            print("No predictions available. Run evaluation first.")
            return None
        
        all_preds = []
        all_gts = []
        
        for pred, gt in zip(self.results['predictions'], self.results['ground_truths']):
            if pred.ndim == 3:
                pred_classes = np.argmax(pred, axis=0).flatten()
            else:
                pred_classes = (pred > 0.5).astype(int).flatten()
            
            gt_classes = gt.flatten().astype(int)
            
            all_preds.extend(pred_classes)
            all_gts.extend(gt_classes)
        
        kappa = cohen_kappa_score(all_gts, all_preds)
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        return kappa


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        save_results: Whether to save results
        results_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = ModelEvaluator(model, device)
    
    # Run evaluation
    metrics = evaluator.evaluate_dataloader(
        dataloader, 
        save_results=save_results,
        results_dir=results_dir
    )
    
    # Generate visualizations
    evaluator.visualize_predictions(save_path=f"{results_dir}/predictions.png")
    evaluator.plot_confusion_matrix(save_path=f"{results_dir}/confusion_matrix.png")
    evaluator.compute_cohen_kappa()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities for prostate cancer segmentation")
    print("Import this module to use evaluation functions")
    
    # Example of creating metrics calculator
    metrics = SegmentationMetrics(num_classes=6)
    
    # Create dummy data for testing
    pred = torch.rand(2, 6, 256, 256)  # Batch of 2, 6 classes, 256x256
    target = torch.randint(0, 6, (2, 256, 256))  # Ground truth
    
    # Compute metrics
    class_metrics = metrics.class_wise_metrics(pred, target)
    print("Example metrics computed successfully")