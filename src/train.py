   """
Training utilities for prostate cancer semantic segmentation.

This module provides training functions and classes for training
SAM-based models on prostate histopathology data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from monai.losses import DiceCELoss
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import time
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import save_model_checkpoint, format_time, get_device
from .evaluate import SegmentationMetrics


class SAMTrainer:
    """
    Trainer class for SAM-based prostate cancer segmentation.
    
    Handles training loop, validation, checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=5e-5,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = DiceCELoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True
            )
        else:
            self.loss_fn = loss_fn
        
        # Setup metrics
        self.metrics = SegmentationMetrics(num_classes=6)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rates': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"SAMTrainer initialized")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"Validation samples: {len(val_dataloader.dataset)}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        epoch_losses = []
        epoch_dice = []
        epoch_iou = []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f'Epoch {epoch+1} [Train]',
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_boxes = batch['input_boxes'].to(self.device)
                ground_truth_masks = batch['ground_truth_mask'].float().to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                
                # Get predictions
                pred_masks = outputs.pred_masks.squeeze(1)
                
                # Compute loss
                loss = self.loss_fn(pred_masks, ground_truth_masks.unsqueeze(1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    pred_sigmoid = torch.sigmoid(pred_masks)
                    dice_score = self.metrics.dice_coefficient(pred_sigmoid, ground_truth_masks)
                    iou_score = self.metrics.iou_score(pred_sigmoid, ground_truth_masks)
                
                # Store metrics
                epoch_losses.append(loss.item())
                epoch_dice.append(dice_score.item())
                epoch_iou.append(iou_score.item())
                
                # Update progress bar
                if batch_idx % self.log_interval == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': f'{dice_score.item():.4f}',
                        'IoU': f'{iou_score.item():.4f}'
                    })
            
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice) if epoch_dice else 0.0
        avg_iou = np.mean(epoch_iou) if epoch_iou else 0.0
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        epoch_losses = []
        epoch_dice = []
        epoch_iou = []
        
        progress_bar = tqdm(
            self.val_dataloader,
            desc=f'Epoch {epoch+1} [Val]',
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_boxes = batch['input_boxes'].to(self.device)
                    ground_truth_masks = batch['ground_truth_mask'].float().to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_boxes=input_boxes,
                        multimask_output=False
                    )
                    
                    # Get predictions
                    pred_masks = outputs.pred_masks.squeeze(1)
                    
                    # Compute loss
                    loss = self.loss_fn(pred_masks, ground_truth_masks.unsqueeze(1))
                    
                    # Compute metrics
                    pred_sigmoid = torch.sigmoid(pred_masks)
                    dice_score = self.metrics.dice_coefficient(pred_sigmoid, ground_truth_masks)
                    iou_score = self.metrics.iou_score(pred_sigmoid, ground_truth_masks)
                    
                    # Store metrics
                    epoch_losses.append(loss.item())
                    epoch_dice.append(dice_score.item())
                    epoch_iou.append(iou_score.item())
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': f'{dice_score.item():.4f}',
                        'IoU': f'{iou_score.item():.4f}'
                    })
                
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice) if epoch_dice else 0.0
        avg_iou = np.mean(epoch_iou) if epoch_iou else 0.0
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        }
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        save_every: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 60)
        
        best_val_dice = 0.0
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics and 'loss' in val_metrics:
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_dice'].append(val_metrics['dice'])
                self.history['val_iou'].append(val_metrics['iou'])
            
            # Store learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({format_time(epoch_time)})")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Dice: {train_metrics['dice']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Dice: {val_metrics['dice']:.4f}, "
                      f"IoU: {val_metrics['iou']:.4f}")
                
                # Check for best model
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    epochs_without_improvement = 0
                    
                    if save_best:
                        best_model_path = self.checkpoint_dir / "best_model.pth"
                        save_model_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            val_metrics['loss'],
                            val_metrics,
                            str(best_model_path)
                        )
                        print(f"  New best model saved! (Dice: {best_val_dice:.4f})")
                else:
                    epochs_without_improvement += 1
            
            print(f"  LR: {current_lr:.2e}")
            print("-" * 40)
            
            # Save periodic checkpoint
            if save_every and (epoch + 1) % save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                save_model_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics['loss'],
                    train_metrics,
                    str(checkpoint_path)
                )
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"No improvement for {early_stopping_patience} epochs")
                break
        
        # Save final model
        final_model_path = self.checkpoint_dir / "final_model.pth"
        save_model_checkpoint(
            self.model,
            self.optimizer,
            num_epochs - 1,
            train_metrics['loss'],
            train_metrics,
            str(final_model_path)
        )
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice')
        if self.history['val_dice']:
            axes[0, 1].plot(self.history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        axes[1, 0].plot(self.history['train_iou'], label='Train IoU')
        if self.history['val_iou']:
            axes[1, 0].plot(self.history['val_iou'], label='Val IoU')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    checkpoint_dir: str = "checkpoints"
) -> SAMTrainer:
    """
    Create a configured SAMTrainer instance.
    
    Args:
        model: SAM model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        use_scheduler: Whether to use learning rate scheduler
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        Configured SAMTrainer
    """
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    # Setup loss function
    loss_fn = DiceCELoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True
    )
    
    # Create trainer
    trainer = SAMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer


def train_sam_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    checkpoint_dir: str = "checkpoints",
    save_best: bool = True,
    early_stopping_patience: Optional[int] = 10
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a SAM model with default configuration.
    
    Args:
        model: SAM model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        save_best: Whether to save best model
        early_stopping_patience: Early stopping patience
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train model
    history = trainer.train(
        num_epochs=num_epochs,
        save_best=save_best,
        early_stopping_patience=early_stopping_patience
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=f"{checkpoint_dir}/training_history.png"
    )
    
    return model, history


if __name__ == "__main__":
    print("Training utilities for prostate cancer segmentation")
    print("Use create_trainer() or train_sam_model() for training")