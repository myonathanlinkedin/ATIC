"""
Inference Time Chart Module
Generates inference time comparison between SNN and ANN models.
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


class InferenceTimeChart(BaseChart):
    """Inference time comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate inference time comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Extract inference time data (strictly from results)
            snn_inference = results.get('snn', {}).get('inference_time', None)
            ann_inference = results.get('ann', {}).get('inference_time', None)

            # If lists are provided, take the last value; if missing, use 0.0
            if isinstance(snn_inference, list):
                snn_inference = (
                    snn_inference[-1] if len(snn_inference) > 0 else 0.0
                )
            if isinstance(ann_inference, list):
                ann_inference = (
                    ann_inference[-1] if len(ann_inference) > 0 else 0.0
                )
            snn_inference = (
                float(snn_inference) if snn_inference is not None else 0.0
            )
            ann_inference = (
                float(ann_inference) if ann_inference is not None else 0.0
            )

            # If both are zero/missing, fall back to a neutral placeholder chart
            if snn_inference == 0.0 and ann_inference == 0.0:
                self.create_fallback_chart('04_inference_time', 'Inference Time Comparison')
                return
            
            # Create bar chart
            _, ax = plt.subplots(figsize=(10, 8))
            models = ['SNN', 'ANN']
            inference_times = [snn_inference, ann_inference]
            colors = ['red', 'blue']
            
            bars = ax.bar(models, inference_times, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, inference_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{time_val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Model Type', fontsize=14)
            ax.set_ylabel('Inference Time (seconds)', fontsize=14)
            ratio = (snn_inference / ann_inference) if ann_inference > 0 else 0.0
            ax.set_title(
                'Inference Time Comparison: SNN vs ANN\n'
                f'SNN is {ratio:.2f}x {"slower" if ratio > 1 else "faster"} than ANN',
                fontsize=14, fontweight='bold'  # Reduced from 16 to 14
            )
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits to show the performance difference clearly
            ax.set_ylim(
                0,
                max(inference_times) * 1.2 if max(inference_times) > 0 else 1.0,
            )
            
            # Add performance context
            perf_text = (
                f'SNN: {snn_inference:.1f}s | ANN: {ann_inference:.1f}s\n'
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
            # Standardized filename to keep total charts at 34
            self.save_chart('04_inference_time')
        except Exception as e:
            print(f"❌ Error in inference time chart: {e}")
            # Standardized fallback filename
            self.create_fallback_chart('04_inference_time', 'Inference Time Comparison')
