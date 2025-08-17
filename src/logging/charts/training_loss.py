"""
Training Loss Chart Module
Generates training loss comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import BaseChart directly
from base_chart import BaseChart


class TrainingLossChart(BaseChart):
    """Training loss comparison chart"""
    
    def extract_data(self, results_data):
        """Extract training loss data from benchmark results"""
        if not results_data:
            return [], []
        
        # Extract training loss from the correct structure
        training_data = []
        epochs = []
        
        # Handle different result formats
        if 'training_results' in results_data:
            # New format with training_results
            for result in results_data['training_results']:
                if 'epoch' in result and 'train_loss' in result:
                    epochs.append(result['epoch'])
                    training_data.append(result['train_loss'])
        elif 'training_loss' in results_data:
            # Direct array format
            training_data = results_data['training_loss']
            epochs = list(range(1, len(training_data) + 1))
        elif 'results' in results_data:
            # Legacy format
            for result in results_data['results']:
                if 'epoch' in result and 'train_loss' in result:
                    epochs.append(result['epoch'])
                    training_data.append(result['train_loss'])
        else:
            # Fallback: try to find any loss data
            for key, value in results_data.items():
                if isinstance(value, list) and len(value) > 0:
                    for item in value:
                        if isinstance(item, dict) and 'train_loss' in item:
                            if 'epoch' in item:
                                epochs.append(item['epoch'])
                            else:
                                epochs.append(len(epochs))
                            training_data.append(item['train_loss'])
        
        # Ensure we have valid data
        if not training_data:
            print("⚠️  No training loss data found in results")
            return [], []
        
        return epochs, training_data
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract training loss data with dynamic epoch support
            snn_train_loss = self.safe_extract_list_data(
                results.get('snn', {}), 'train_losses')
            ann_train_loss = self.safe_extract_list_data(
                results.get('ann', {}), 'train_losses')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_train_loss and not ann_train_loss:
                print(f"⚠️  Warning: No training loss data found")
                self.create_fallback_chart('03_training_loss', 'Training Loss Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_train_loss) if snn_train_loss else 0,
                len(ann_train_loss) if ann_train_loss else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('03_training_loss', 'Training Loss Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN training loss
            if snn_train_loss and len(snn_train_loss) > 0:
                snn_plot_data = snn_train_loss[:max_length] if len(snn_train_loss) > max_length else snn_train_loss
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Training Loss', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN training loss
            if ann_train_loss and len(ann_train_loss) > 0:
                ann_plot_data = ann_train_loss[:max_length] if len(ann_train_loss) > max_length else ann_train_loss
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Training Loss', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Training Loss', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            # Single title with explanatory text to avoid duplication
            fig.suptitle('Training Loss Progression: SNN vs ANN\nLower Loss Indicates Better Learning',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_chart('03_training_loss')
            
        except Exception as e:
            print(f"❌ Error in training loss chart: {e}")
            self.create_fallback_chart('03_training_loss', 'Training Loss Analysis')
    
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
