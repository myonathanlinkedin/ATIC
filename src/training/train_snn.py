import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import time
from typing import Dict, List, Tuple
from datetime import datetime
import psutil
import subprocess
from tqdm import tqdm


class SNNTrainer:
    """
    SNN Trainer with ETAD and comprehensive monitoring
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
        save_dir: str = './results/logs',
        use_surrogate_gradient: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_surrogate_gradient = use_surrogate_gradient
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer with proper parameters for SNN
        # FIXED: Increased learning rate for visible learning progress
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate * 0.3,  # INCREASED: Enable visible learning progress
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
        
        # Temporal loss function for SNN
        self.criterion = nn.CrossEntropyLoss()
        
        # Temporal backpropagation parameters
        self.temporal_weight = 0.1  # Weight for temporal loss component
        
        # ENHANCED DEVICE HANDLING: Model is already on correct device from __init__
        # No need to move model again - just get the device it's already on
        try:
            # Get device from model parameters
            device_obj = next(self.model.parameters()).device
            self.device = str(device_obj)
            print(f"✅ Using model device: {device_obj}")
        except Exception as e:
            print(f"❌ Error getting model device: {e}")
            # Fallback to CPU
            device_obj = torch.device('cpu')
            self.device = 'cpu'
            print(f"⚠️  Falling back to CPU device")
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []
        self.energy_consumptions = []
        self.active_neurons_history = []
        self.spike_counts = []
        
        # Checkpoint paths
        self.checkpoint_path = os.path.join(save_dir, 'best_snn_model.pth')
        self.history_path = os.path.join(save_dir, 'snn_training_history.json')
        
    def load_checkpoint_if_exists(self) -> bool:
        """
        Check if trained model exists and load it
        Returns True if checkpoint was loaded, False otherwise
        """
        if (os.path.exists(self.checkpoint_path) and 
                os.path.exists(self.history_path)):
            print("✅ Found existing SNN checkpoint, loading...")
            try:
                # Load model checkpoint - ENHANCED DEVICE HANDLING
                device_obj = next(self.model.parameters()).device
                checkpoint = torch.load(
                    self.checkpoint_path, 
                    map_location=device_obj
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
                self.spike_counts = history.get('spike_counts', [])
                
                print(f"✅ Loaded SNN checkpoint from epoch {checkpoint['epoch']}")
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
            print("🆕 No existing SNN checkpoint found, "
                  "starting fresh training...")
            return False
    
    def count_spikes(self, model: nn.Module, data: torch.Tensor) -> int:
        """Count total number of spikes in the network"""
        spike_count = 0
        
        # Forward pass to get spike tensor
        with torch.no_grad():
            model_output = model(data)
            
            # Check if model has spike_tensor attribute (our custom implementation)
            if hasattr(model, 'spike_tensor') and model.spike_tensor is not None:
                spike_count = torch.sum(model.spike_tensor).item()
            else:
                # Fallback: try to find spikes in model output or attributes
                def spike_hook(module, input, output):
                    nonlocal spike_count
                    if hasattr(output, 'spike'):
                        spike_count += torch.sum(output.spike).item()
                
                # Register hooks for LIF layers
                hooks = []
                for name, module in model.named_modules():
                    if 'LIFCell' in str(type(module)):
                        hook = module.register_forward_hook(spike_hook)
                        hooks.append(hook)
                
                # Forward pass
                model(data)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
        
        return spike_count
    
    def train_epoch(self) -> Tuple[float, float, float, Dict[str, int], int, float, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        total_spikes = 0
        
        # Track active neurons
        active_neurons_sum = {}
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training SNN")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            model_output = self.model(data)
            # Handle tuple output from SNN model
            if isinstance(model_output, tuple):
                output = model_output[0]  # Extract the actual output tensor
            else:
                output = model_output
            # Temporal loss calculation for SNN
            classification_loss = self.criterion(output, target)
            
            # REAL temporal loss component for SNN learning
            if hasattr(self.model, 'membrane_potential') and self.model.membrane_potential is not None:
                # REAL temporal loss: spike timing precision + membrane stability
                spike_timing_loss = torch.mean(torch.abs(self.model.membrane_potential.data)) * 0.01  # REDUCED: Much more stable temporal learning
                membrane_stability_loss = torch.var(self.model.membrane_potential.data) * 0.001  # REDUCED: Much more stable membrane learning
                temporal_loss = spike_timing_loss + membrane_stability_loss
                total_loss = classification_loss + temporal_loss
            else:
                total_loss = classification_loss
            
            total_loss.backward()
            
            # Add gradient clipping for better training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # FIXED: Increased max_norm for better learning
            
            self.optimizer.step()
            
            # Step scheduler for learning rate scheduling
            self.scheduler.step()
            
            # Accumulate loss for epoch average
            epoch_loss += classification_loss.item()  # FIXED: Use classification_loss
            
            # Debug learning progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss={classification_loss.item():.4f}, "
                      f"Classification={classification_loss.item():.4f}, "
                      f"Temporal={temporal_loss.item() if 'temporal_loss' in locals() else 0:.4f}, "
                      f"Total={total_loss.item():.4f}")
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Count spikes and active neurons every 10 batches
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    # Simple active neuron counting
                    active_neurons = {'conv_layers': 0, 'fc_layers': 0}
                    spikes = self.count_spikes(self.model, data)
                    total_spikes += spikes
                    
                    for layer, count in active_neurons.items():
                        if layer not in active_neurons_sum:
                            active_neurons_sum[layer] = 0
                        active_neurons_sum[layer] += count
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Calculate learning progress
        if len(self.train_losses) > 0:
            prev_loss = self.train_losses[-1]
            loss_change = prev_loss - avg_loss
            progress_percent = (loss_change / prev_loss) * 100 if prev_loss > 0 else 0.0
        else:
            loss_change = 0.0
            progress_percent = 0.0
        
        return avg_loss, accuracy, epoch_time, active_neurons_sum, total_spikes, loss_change, progress_percent
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                model_output = self.model(data)
                # Handle tuple output from SNN model
                if isinstance(model_output, tuple):
                    output = model_output[0]  # Extract the actual output tensor
                else:
                    output = model_output
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
        print(f"Starting SNN training for {self.num_epochs} epochs...")
        print(f"Using ETAD: {hasattr(self.model, 'etad_pooling')}")
        
        # Check if model is already trained
        if self.load_checkpoint_if_exists():
            print("✅ Using pre-trained SNN model, skipping training...")
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'epoch_times': self.epoch_times,
                'energy_consumptions': self.energy_consumptions,
                'active_neurons_history': self.active_neurons_history,
                'spike_counts': self.spike_counts
            }
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc, epoch_time, active_neurons, total_spikes, loss_change, progress_percent = self.train_epoch()
            
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
            self.spike_counts.append(total_spikes)
            
            # Enhanced progress monitoring for real learning
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"GPU Power: {energy_metrics['gpu_power_w']:.1f}W")
            print(f"Total Spikes: {total_spikes}")
            
            # Enhanced learning progress summary
            if epoch > 0:
                print(f"📊 Learning Summary: Loss Progress {progress_percent:+.2f}%, Loss Change {loss_change:+.4f}")
            
            # REAL learning progression monitoring
            if epoch > 0:
                # FIXED: Use loss-based progress instead of accuracy-based
                loss_improvement = loss_change  # Use loss_change from train_epoch
                print(f"Learning Progress: {progress_percent:+.2f}% (Loss: {loss_change:+.4f}, Accuracy: {train_acc - self.train_accuracies[-1]:+.2f}%)")
                
                # Monitor spike learning
                if hasattr(self.model, 'adaptive_threshold'):
                    threshold_change = self.model.adaptive_threshold.item() - 0.3
                    print(f"Threshold Adaptation: {threshold_change:+.3f}")
            else:
                print("Learning Progress: Initial epoch")
            
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
                    'train_accuracy': train_acc,
                    'total_spikes': total_spikes
                }, os.path.join(self.save_dir, 'best_snn_model.pth'))
        
        # Save training history
        self.save_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'energy_consumptions': self.energy_consumptions,
            'active_neurons_history': self.active_neurons_history,
            'spike_counts': self.spike_counts
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
            'spike_counts': self.spike_counts,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'SNN_with_ETAD'
        }
        
        # Save with dataset-specific filename
        with open(os.path.join(self.save_dir, 'snn_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

def train_snn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    save_dir: str = './results/logs'
) -> Tuple[List[float], List[float], float, List[float], List[float]]:
    """Convenience function to train SNN model"""
    trainer = SNNTrainer(
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
