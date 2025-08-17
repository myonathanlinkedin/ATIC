"""
BPI Analysis Chart Module
Generates Biological Plausibility Index analysis comparison between SNN and ANN models.
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


class BpiAnalysisChart(BaseChart):
    """Biological Plausibility Index analysis chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract BPI data with dynamic epoch support
            snn_bpi = self.safe_extract_list_data(
                results.get('snn', {}), 'bpi_over_time')
            ann_bpi = self.safe_extract_list_data(
                results.get('ann', {}), 'bpi_over_time')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_bpi and not ann_bpi:
                print(f"⚠️  Warning: No BPI data found")
                self.create_fallback_chart('15_bpi_analysis', 'BPI Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_bpi) if snn_bpi else 0,
                len(ann_bpi) if ann_bpi else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('15_bpi_analysis', 'BPI Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN BPI
            if snn_bpi and len(snn_bpi) > 0:
                snn_plot_data = snn_bpi[:max_length] if len(snn_bpi) > max_length else snn_bpi
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN BPI', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN BPI
            if ann_bpi and len(ann_bpi) > 0:
                ann_plot_data = ann_bpi[:max_length] if len(ann_bpi) > max_length else ann_bpi
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN BPI', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Biological Plausibility Index (BPI)', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            # Single title with explanatory text to avoid duplication
            fig.suptitle('Biological Plausibility Index (BPI): SNN vs ANN\nHigher Values Indicate Better Biological Plausibility',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_chart('15_bpi_analysis')
            
        except Exception as e:
            print(f"❌ Error in BPI analysis chart: {e}")
            self.create_fallback_chart('15_bpi_analysis', 'BPI Analysis')
    
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
