"""
Memory Usage Chart Module
Generates memory usage comparison between SNN and ANN models.
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


class MemoryUsageChart(BaseChart):
    """Memory usage comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract memory usage data with dynamic epoch support
            snn_memory = self.safe_extract_list_data(
                results.get('snn', {}), 'memory_usage')
            ann_memory = self.safe_extract_list_data(
                results.get('ann', {}), 'memory_usage')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_memory and not ann_memory:
                print(f"⚠️  Warning: No memory usage data found")
                self.create_fallback_chart('05_memory_usage', 'Memory Usage Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_memory) if snn_memory else 0,
                len(ann_memory) if ann_memory else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('05_memory_usage', 'Memory Usage Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN memory usage
            if snn_memory and len(snn_memory) > 0:
                snn_plot_data = snn_memory[:max_length] if len(snn_memory) > max_length else snn_memory
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Memory Usage', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN memory usage
            if ann_memory and len(ann_memory) > 0:
                ann_plot_data = ann_memory[:max_length] if len(ann_memory) > max_length else ann_memory
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Memory Usage', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Memory Usage (MB)', fontsize=14)
            # Remove redundant title, keep only main title
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            # Single clear title without redundancy
            fig.suptitle('Memory Usage Comparison: SNN vs ANN\nLower Memory Indicates Better Efficiency',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_chart('05_memory_usage')
            
        except Exception as e:
            print(f"❌ Error in memory usage chart: {e}")
            self.create_fallback_chart('05_memory_usage', 'Memory Usage Analysis')
    
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
