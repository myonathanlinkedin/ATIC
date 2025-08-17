"""
Neural Synchronization Chart Module
Generates neural synchronization over time line chart for SNN and ANN models.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import sys
import os
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)
from base_chart import BaseChart
class NeuralSynchronizationChart(BaseChart):
    """Neural synchronization over time chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract neural synchronization data with dynamic epoch support
            snn_sync = self.safe_extract_list_data(
                results.get('snn', {}), 'neural_synchronization')
            ann_sync = self.safe_extract_list_data(
                results.get('ann', {}), 'neural_synchronization')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_sync and not ann_sync:
                print(f"⚠️  Warning: No neural synchronization data found")
                self.create_fallback_chart('19_neural_synchronization', 'Neural Synchronization Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_sync) if snn_sync else 0,
                len(ann_sync) if ann_sync else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('19_neural_synchronization', 'Neural Synchronization Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN neural synchronization
            if snn_sync and len(snn_sync) > 0:
                snn_plot_data = snn_sync[:max_length] if len(snn_sync) > max_length else snn_sync
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Neural Synchronization', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN neural synchronization
            if ann_sync and len(ann_sync) > 0:
                ann_plot_data = ann_sync[:max_length] if len(ann_sync) > max_length else ann_sync
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Neural Synchronization', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Neural Synchronization Index', fontsize=14)
            ax.set_title('Neural Synchronization: SNN vs ANN\nHigher Values Indicate Better Synchronization', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            plt.tight_layout()
            self.save_chart('19_neural_synchronization')
            
        except Exception as e:
            print(f"❌ Error in neural synchronization chart: {e}")
            self.create_fallback_chart('19_neural_synchronization', 'Neural Synchronization Analysis')
    
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
