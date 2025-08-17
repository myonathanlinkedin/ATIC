"""
GPU Power Chart Module
Generates GPU power consumption comparison between SNN and ANN models.
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


class GpuPowerChart(BaseChart):
    """GPU power consumption comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract GPU power data with dynamic epoch support
            snn_gpu_power = self.safe_extract_list_data(
                results.get('snn', {}), 'gpu_power')
            ann_gpu_power = self.safe_extract_list_data(
                results.get('ann', {}), 'gpu_power')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_gpu_power and not ann_gpu_power:
                print(f"⚠️  Warning: No GPU power data found")
                self.create_fallback_chart('07_gpu_power', 'GPU Power Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_gpu_power) if snn_gpu_power else 0,
                len(ann_gpu_power) if ann_gpu_power else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('07_gpu_power', 'GPU Power Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN GPU power
            if snn_gpu_power and len(snn_gpu_power) > 0:
                snn_plot_data = snn_gpu_power[:max_length] if len(snn_gpu_power) > max_length else snn_gpu_power
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN GPU Power', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN GPU power
            if ann_gpu_power and len(ann_gpu_power) > 0:
                ann_plot_data = ann_gpu_power[:max_length] if len(ann_gpu_power) > max_length else ann_gpu_power
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN GPU Power', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('GPU Power (Watts)', fontsize=14)
            ax.set_title('GPU Power Consumption: SNN vs ANN\nLower Power Indicates Better Energy Efficiency', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            plt.tight_layout()
            self.save_chart('07_gpu_power')
            
        except Exception as e:
            print(f"❌ Error in GPU power chart: {e}")
            self.create_fallback_chart('07_gpu_power', 'GPU Power Analysis')
    
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
