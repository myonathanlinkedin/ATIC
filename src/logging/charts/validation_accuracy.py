"""
Validation Accuracy Chart Module
Generates validation accuracy comparison between SNN and ANN models.
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


class ValidationAccuracyChart(BaseChart):
    """Validation accuracy comparison chart"""
    
    def extract_data(self, results_data):
        """Extract validation accuracy data from benchmark results"""
        if not results_data:
            return [], []
        
        # Extract validation accuracy from the correct structure
        validation_data = []
        epochs = []
        
        # Handle comprehensive benchmark results structure
        if 'snn' in results_data and 'ann' in results_data:
            # New comprehensive format: results_data['snn']['validation_accuracy']
            snn_data = results_data.get('snn', {}).get(
                'validation_accuracy', [])
            ann_data = results_data.get('ann', {}).get(
                'validation_accuracy', [])
            
            if snn_data and ann_data:
                # Use SNN data for epochs (both should have same length)
                epochs = list(range(1, len(snn_data) + 1))
                validation_data = snn_data  # Return SNN data for this chart
                return epochs, validation_data
        
        # Handle direct training_results format
        if 'training_results' in results_data:
            for result in results_data['training_results']:
                if 'epoch' in result and 'val_accuracy' in result:
                    epochs.append(result['epoch'])
                    validation_data.append(result['val_accuracy'])
        
        # Handle direct array format
        elif 'validation_accuracy' in results_data:
            validation_data = results_data['validation_accuracy']
            epochs = list(range(1, len(validation_data) + 1))
        
        # Handle legacy format
        elif 'results' in results_data:
            for result in results_data['results']:
                if 'epoch' in result and 'val_accuracy' in result:
                    epochs.append(result['epoch'])
                    validation_data.append(result['val_accuracy'])
        
        # Ensure we have valid data
        if not validation_data:
            print("⚠️  No validation accuracy data found in results")
            return [], []
        
        return epochs, validation_data
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract validation accuracy data with dynamic epoch support
            snn_val_acc = self.safe_extract_list_data(
                results.get('snn', {}), 'val_accuracies')
            ann_val_acc = self.safe_extract_list_data(
                results.get('ann', {}), 'val_accuracies')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_val_acc and not ann_val_acc:
                print(f"⚠️  Warning: No validation accuracy data found")
                self.create_fallback_chart('02_validation_accuracy', 'Validation Accuracy Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_val_acc) if snn_val_acc else 0,
                len(ann_val_acc) if ann_val_acc else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('02_validation_accuracy', 'Validation Accuracy Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN validation accuracy
            if snn_val_acc and len(snn_val_acc) > 0:
                snn_plot_data = snn_val_acc[:max_length] if len(snn_val_acc) > max_length else snn_val_acc
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Validation Accuracy', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN validation accuracy
            if ann_val_acc and len(ann_val_acc) > 0:
                ann_plot_data = ann_val_acc[:max_length] if len(ann_val_acc) > max_length else ann_val_acc
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Validation Accuracy', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            # Single title with explanatory text to avoid duplication
            fig.suptitle('Validation Accuracy Progression: SNN vs ANN\nHigher Accuracy Indicates Better Generalization',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_chart('02_validation_accuracy')
            
        except Exception as e:
            print(f"❌ Error in validation accuracy chart: {e}")
            self.create_fallback_chart('02_validation_accuracy', 'Validation Accuracy Analysis')
    
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
