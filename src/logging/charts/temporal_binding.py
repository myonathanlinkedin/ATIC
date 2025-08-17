"""
Temporal Binding Chart Module
Generates temporal binding over time line chart for SNN and ANN models.
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
class TemporalBindingChart(BaseChart):
    """Temporal binding over time chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract temporal binding data with dynamic epoch support
            snn_binding = self.safe_extract_list_data(
                results.get('snn', {}), 'temporal_binding')
            ann_binding = self.safe_extract_list_data(
                results.get('ann', {}), 'temporal_binding')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_binding and not ann_binding:
                print(f"⚠️  Warning: No temporal binding data found")
                self.create_fallback_chart('25_temporal_binding', 'Temporal Binding Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_binding) if snn_binding else 0,
                len(ann_binding) if ann_binding else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('25_temporal_binding', 'Temporal Binding Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN temporal binding
            if snn_binding and len(snn_binding) > 0:
                snn_plot_data = snn_binding[:max_length] if len(snn_binding) > max_length else snn_binding
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Temporal Binding', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN temporal binding
            if ann_binding and len(ann_binding) > 0:
                ann_plot_data = ann_binding[:max_length] if len(ann_binding) > max_length else ann_binding
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Temporal Binding', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Temporal Binding Index', fontsize=14)
            ax.set_title('Temporal Binding: SNN vs ANN\nHigher Values Indicate Better Temporal Coherence', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            # Reduce main title font size
            fig.suptitle('Temporal Binding Analysis: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            
            plt.tight_layout()
            self.save_chart('25_temporal_binding')
            
        except Exception as e:
            print(f"❌ Error in temporal binding chart: {e}")
            self.create_fallback_chart('25_temporal_binding', 'Temporal Binding Analysis')
    
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
