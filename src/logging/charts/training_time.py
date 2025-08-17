"""
Training Time Chart Module
Generates training time comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import BaseChart with package-safe fallback
try:  # pragma: no cover
    from .base_chart import BaseChart  # type: ignore
except Exception:  # pragma: no cover
    from base_chart import BaseChart  # type: ignore


class TrainingTimeChart(BaseChart):
    """Training time comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate training time comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Extract training time data (strictly from results)
            snn_training = results.get('snn', {}).get('training_time', None)
            ann_training = results.get('ann', {}).get('training_time', None)

            # If lists are provided, use the last available value; if missing → 0.0
            if isinstance(snn_training, list):
                snn_training = float(snn_training[-1]) if len(snn_training) > 0 else 0.0
            if isinstance(ann_training, list):
                ann_training = float(ann_training[-1]) if len(ann_training) > 0 else 0.0
            snn_training = float(snn_training) if snn_training is not None else 0.0
            ann_training = float(ann_training) if ann_training is not None else 0.0

            # If both are zero/missing, render neutral fallback
            if snn_training == 0.0 and ann_training == 0.0:
                self.create_fallback_chart('09_training_time', 'Training Time Comparison')
                return
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            models = ['SNN', 'ANN']
            training_times = [snn_training, ann_training]
            colors = ['red', 'blue']
            
            bars = ax.bar(models, training_times, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, training_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{time_val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Model Type', fontsize=14)
            ax.set_ylabel('Training Time (seconds)', fontsize=14)
            ratio = (snn_training / ann_training) if ann_training > 0 else 0.0
            
            # Reduce main title font size and fix title
            ax.set_title(
                'Training Time Comparison: SNN vs ANN\n'
                'Lower values indicate faster training\n'
                f'Observed ratio (SNN/ANN): {ratio:.2f}x',
                fontsize=14,  # Reduced from 16 to 14
                fontweight='bold',
            )
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits to show the performance difference clearly
            ax.set_ylim(0, max(training_times) * 1.2 if max(training_times) > 0 else 1.0)
            
            # Add performance context
            perf_text = (
                f'SNN: {snn_training:.1f}s | ANN: {ann_training:.1f}s\n'
                f'Ratio: SNN/ANN = {ratio:.1f}x'
            )
            ax.text(
                0.02,
                0.95,
                perf_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            )
            
            plt.tight_layout()
            self.save_chart('09_training_time')
        except Exception as e:
            print(f"❌ Error in training time chart: {e}")
            self.create_fallback_chart('09_training_time', 'Training Time Comparison')
