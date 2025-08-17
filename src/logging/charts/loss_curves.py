"""
Loss Curves Chart Module
Generates training and validation loss curves comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import sys
import os

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import BaseChart directly
from base_chart import BaseChart


class LossCurvesChart(BaseChart):
    """Loss curves comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract loss data with dynamic epoch support
            snn_train_losses = self.safe_extract_list_data(
                results.get('snn', {}), 'train_losses')
            snn_val_losses = self.safe_extract_list_data(
                results.get('snn', {}), 'val_losses')
            ann_train_losses = self.safe_extract_list_data(
                results.get('ann', {}), 'train_losses')
            ann_val_losses = self.safe_extract_list_data(
                results.get('ann', {}), 'val_losses')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not any([snn_train_losses, snn_val_losses, ann_train_losses, ann_val_losses]):
                print(f"⚠️  Warning: No loss data found")
                self.create_fallback_chart('12_loss_curves', 'Loss Curves Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_train_losses) if snn_train_losses else 0,
                len(snn_val_losses) if snn_val_losses else 0,
                len(ann_train_losses) if ann_train_losses else 0,
                len(ann_val_losses) if ann_val_losses else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('12_loss_curves', 'Loss Curves Analysis')
                return
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Chart 1: Training Losses
            ax1.set_title('Training Losses: SNN vs ANN', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Epochs', fontsize=12)
            ax1.set_ylabel('Training Loss', fontsize=12)
            
            if snn_train_losses and len(snn_train_losses) > 0:
                snn_plot_data = snn_train_losses[:max_length] if len(snn_train_losses) > max_length else snn_train_losses
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax1.plot(snn_time_steps, snn_plot_data, 'o-', 
                        label='SNN Training Loss', linewidth=2, markersize=6, color='lightblue')
            
            if ann_train_losses and len(ann_train_losses) > 0:
                ann_plot_data = ann_train_losses[:max_length] if len(ann_train_losses) > max_length else ann_train_losses
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax1.plot(ann_time_steps, ann_plot_data, 's-', 
                        label='ANN Training Loss', linewidth=2, markersize=6, color='lightcoral')
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1, max_length)
            
            # Chart 2: Validation Losses
            ax2.set_title('Validation Losses: SNN vs ANN', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Training Epochs', fontsize=12)
            ax2.set_ylabel('Validation Loss', fontsize=12)
            
            if snn_val_losses and len(snn_val_losses) > 0:
                snn_plot_data = snn_val_losses[:max_length] if len(snn_val_losses) > max_length else snn_val_losses
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax2.plot(snn_time_steps, snn_plot_data, 'o-', 
                        label='SNN Validation Loss', linewidth=2, markersize=6, color='lightblue')
            
            if ann_val_losses and len(ann_val_losses) > 0:
                ann_plot_data = ann_val_losses[:max_length] if len(ann_val_losses) > max_length else ann_val_losses
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax2.plot(ann_time_steps, ann_plot_data, 's-', 
                        label='ANN Validation Loss', linewidth=2, markersize=6, color='lightcoral')
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(1, max_length)
            
            plt.tight_layout()
            self.save_chart('12_loss_curves')
            
        except Exception as e:
            print(f"❌ Error in loss curves chart: {e}")
            self.create_fallback_chart('12_loss_curves', 'Loss Curves Analysis')
    
    def _detect_epoch_length(self, results: Dict[str, Any]) -> int:
        """Auto-detect epoch length from available data"""
        try:
            # Try to get epoch length from training data
            training_keys = ['training_accuracy', 'training_loss', 'validation_accuracy', 'validation_loss']
            for key in training_keys:
                if key in results:
                    data = results[key]
                    if isinstance(data, list) and len(data) > 0:
                        return len(data)
            
            # Try to get from SNN results
            if 'snn' in results and isinstance(results['snn'], dict):
                for key in training_keys:
                    if key in results['snn']:
                        data = results['snn'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            # Try to get from ANN results
            if 'ann' in results and isinstance(results['ann'], dict):
                for key in training_keys:
                    if key in results['ann']:
                        data = results['ann'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            return 10  # Default fallback
            
        except Exception:
            return 10 
