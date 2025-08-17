"""
ATIC Sensitivity Chart Module
Generates ATIC sensitivity over time line chart for SNN and ANN models.
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
class AticSensitivityChart(BaseChart):
    """ATIC sensitivity over time chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            # Extract ATIC sensitivity data
            snn_sensitivity = results.get('snn', {}).get('atic_sensitivity', [])
            ann_sensitivity = results.get('ann', {}).get('atic_sensitivity', [])
            
            # Use actual data length, no hard-coded limits
            if not snn_sensitivity and not ann_sensitivity:
                print(f"⚠️  Warning: No ATIC sensitivity data found")
                self.create_fallback_chart('23_atic_sensitivity', 'ATIC Sensitivity Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_sensitivity) if snn_sensitivity else 0,
                len(ann_sensitivity) if ann_sensitivity else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('23_atic_sensitivity', 'ATIC Sensitivity Analysis')
                return
            
            time_steps = range(1, max_length + 1)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN ATIC sensitivity
            if snn_sensitivity and len(snn_sensitivity) > 0:
                # Use actual data length, no padding to fixed size
                snn_plot_data = snn_sensitivity[:max_length] if len(snn_sensitivity) > max_length else snn_sensitivity
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN ATIC Sensitivity', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN ATIC sensitivity
            if ann_sensitivity and len(ann_sensitivity) > 0:
                # Use actual data length, no padding to fixed size
                ann_plot_data = ann_sensitivity[:max_length] if len(ann_sensitivity) > max_length else ann_sensitivity
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN ATIC Sensitivity', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('ATIC Sensitivity', fontsize=14)
            ax.set_title('ATIC Sensitivity Analysis: SNN vs ANN\nAdaptive Temporal Information Compression', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            plt.tight_layout()
            self.save_chart('23_atic_sensitivity')
            
        except Exception as e:
            print(f"❌ Error in ATIC sensitivity chart: {e}")
            self.create_fallback_chart('23_atic_sensitivity', 'ATIC Sensitivity Analysis')
    
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
