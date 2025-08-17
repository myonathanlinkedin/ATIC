import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime

from ..models.ann_baseline import ANNBaseline, ANNBaselineSHD
from ..models.utils import set_seed, count_active_neurons, calculate_energy_efficiency

class ANNTrainer:
    """
    ANN Trainer with comprehensive monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 20,
        save_dir: str = './results/logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer with better learning rate for ANN
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate * 1.5,  # Increased learning rate for better ANN learning
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Add learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs * 2,  # FIXED: Increased T_max for better convergence
            eta_min=1e-5  # FIXED: Increased minimum learning rate
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Move model to device - PROVEN SOLUTION
        # Skip device checking to avoid CUDA errors
        self.model = self.model.to(device)
        print(f"✅ ANN model moved to {device}")
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []
        self.energy_consumptions = []
        self.active_neurons_history = []
        
        # Checkpoint paths
        self.checkpoint_path = os.path.join(save_dir, 'best_ann_model.pth')
        self.history_path = os.path.join(save_dir, 'ann_training_history.json')
        
    def load_checkpoint_if_exists(self) -> bool:
        """
        Check if trained model exists and load it
        Returns True if checkpoint was loaded, False otherwise
        """
        if (os.path.exists(self.checkpoint_path) and 
                os.path.exists(self.history_path)):
            print("✅ Found existing ANN checkpoint, loading...")
            try:
                # Load model checkpoint - FIX DEVICE HANDLING
                checkpoint = torch.load(
                    self.checkpoint_path, 
                    map_location=torch.device(self.device)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict']
                )
                
                # Load training history
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
                
                # Restore metrics
                self.train_losses = history.get('train_losses', [])
                self.val_losses = history.get('val_losses', [])
                self.train_accuracies = history.get('train_accuracies', [])
                self.val_accuracies = history.get('val_accuracies', [])
                self.epoch_times = history.get('epoch_times', [])
                self.energy_consumptions = history.get(
                    'energy_consumptions', []
                )
                self.active_neurons_history = history.get(
                    'active_neurons_history', []
                )
                
                print(f"✅ Loaded ANN checkpoint from epoch {checkpoint['epoch']}")
                print(f"✅ Best validation accuracy: "
                      f"{checkpoint['val_accuracy']:.2f}%")
                print(f"✅ Training history restored with "
                      f"{len(self.train_losses)} epochs")
                
                return True
                
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("⚠️  Starting fresh training...")
                return False
        else:
            print("🆕 No existing ANN checkpoint found, "
                  "starting fresh training...")
            return False
    
    def train_epoch(self) -> Tuple[float, float, float, Dict[str, int]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Track active neurons
        active_neurons_sum = {}
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Add gradient clipping for better training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # FIXED: Increased max_norm for better learning
            
            self.optimizer.step()
            
            # Step scheduler for learning rate scheduling
            self.scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Debug learning progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"ANN Batch {batch_idx}: Loss={loss.item():.4f}")
            
            # Count active neurons every 10 batches
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    active_neurons = count_active_neurons(self.model, data)
                    for layer, count in active_neurons.items():
                        if layer not in active_neurons_sum:
                            active_neurons_sum[layer] = 0
                        active_neurons_sum[layer] += count
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, epoch_time, active_neurons_sum
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def measure_energy_consumption(self) -> Dict[str, float]:
        """Measure energy consumption during training"""
        # Get GPU power consumption if available
        gpu_power = 0.0
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                gpu_power = float(result.stdout.strip())
        except:
            pass
        
        # Get CPU power consumption
        cpu_percent = psutil.cpu_percent(interval=1)
        from src.config.constants import HardwareConfig
        cpu_power = cpu_percent * HardwareConfig.DEFAULT_CPU_POWER_ESTIMATE  # Use constant instead of hardcoded value
        
        return {
            'gpu_power_w': gpu_power,
            'cpu_power_w': cpu_power,
            'total_power_w': gpu_power + cpu_power
        }
    
    def train(self) -> Dict[str, List]:
        """Main training loop with checkpoint loading"""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        # Check if model is already trained
        if self.load_checkpoint_if_exists():
            print("✅ Using pre-trained ANN model, skipping training...")
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'epoch_times': self.epoch_times,
                'energy_consumptions': self.energy_consumptions,
                'active_neurons_history': self.active_neurons_history
            }
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc, epoch_time, active_neurons = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Measure energy consumption
            energy_metrics = self.measure_energy_consumption()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.epoch_times.append(epoch_time)
            self.energy_consumptions.append(energy_metrics)
            self.active_neurons_history.append(active_neurons)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"GPU Power: {energy_metrics['gpu_power_w']:.1f}W")
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc
                }, os.path.join(self.save_dir, 'best_ann_model.pth'))
        
        # Save training history
        self.save_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'energy_consumptions': self.energy_consumptions,
            'active_neurons_history': self.active_neurons_history
        }
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'energy_consumptions': self.energy_consumptions,
            'active_neurons_history': self.active_neurons_history,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'ANN_Baseline'
        }
        
        # Save with dataset-specific filename
        with open(os.path.join(self.save_dir, 'ann_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

def train_ann_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    save_dir: str = './results/logs'
) -> Tuple[List[float], List[float], float, List[float], List[float]]:
    """Convenience function to train ANN model"""

    
    trainer = ANNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    history = trainer.train()
    
    # Return the expected tuple format: (train_losses, val_losses, final_accuracy, train_accuracies, val_accuracies)
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    train_accuracies = history.get('train_accuracies', [])
    val_accuracies = history.get('val_accuracies', [])
    final_accuracy = val_accuracies[-1] if val_accuracies else 0.0
    
    return train_losses, val_losses, final_accuracy, train_accuracies, val_accuracies 
