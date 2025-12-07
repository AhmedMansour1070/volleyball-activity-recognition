"""
Training script for Hierarchical Group Activity Recognition
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time

from src.data.volleyball_dataset import VolleyballDataset
from src.models.hierarchical_model import HierarchicalModel
from src.utils.collate_fn import volleyball_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for hierarchical model"""
    
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_scheduler']['step_size'],
            gamma=config['training']['lr_scheduler']['gamma']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['activity_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs['activity_logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs['activity_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            features = batch['features'].to(self.device)
            labels = batch['activity_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs['activity_logits'], labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs['activity_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                logger.info(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
        
        logger.info("Training complete!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = Path('outputs/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/volleyball_config.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override batch size if specified
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = VolleyballDataset(
        video_ids=config['dataset']['train_videos'],
        features_dir=Path('data/volleyball/features/alexnet'),
        config=config['dataset'],
        split='train'
    )
    
    val_dataset = VolleyballDataset(
        video_ids=config['dataset']['val_videos'],
        features_dir=Path('data/volleyball/features/alexnet'),
        config=config['dataset'],
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=volleyball_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=volleyball_collate_fn
    )
    
    # Create model
    logger.info("Creating model...")
    
    model = HierarchicalModel(
        feature_dim=9216,  # AlexNet features
        person_hidden_dim=config['model']['person_lstm']['hidden_dim'],
        group_hidden_dim=config['model']['group_lstm']['hidden_dim'],
        num_action_classes=config['model']['num_action_classes'],
        num_activity_classes=config['model']['num_activity_classes'],
        pooling=config['model']['pooling'],
        dropout=config['model']['person_lstm']['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()