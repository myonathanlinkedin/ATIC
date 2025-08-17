"""
Hardware Utilization Chart Module
Generates hardware utilization patterns comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os
from base_chart import BaseChart

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)


class HardwareUtilizationChart(BaseChart):
    """Hardware utilization patterns comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract hardware utilization data with dynamic epoch support
            snn_gpu_util = self.safe_extract_list_data(results.get('snn', {}), 'gpu_utilization')
            snn_memory_util = self.safe_extract_list_data(results.get('snn', {}), 'memory_utilization')
            ann_gpu_util = self.safe_extract_list_data(results.get('ann', {}), 'gpu_utilization')
            ann_memory_util = self.safe_extract_list_data(results.get('ann', {}), 'memory_utilization')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not any([snn_gpu_util, snn_memory_util, ann_gpu_util, ann_memory_util]):
                print(f"⚠️  Warning: No hardware utilization data found")
                self.create_fallback_chart('14_hardware_utilization', 'Hardware Utilization Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_gpu_util) if snn_gpu_util else 0,
                len(snn_memory_util) if snn_memory_util else 0,
                len(ann_gpu_util) if ann_gpu_util else 0,
                len(ann_memory_util) if ann_memory_util else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('14_hardware_utilization', 'Hardware Utilization Analysis')
                return
            
            time_steps = range(1, max_length + 1)
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Chart 1: GPU Utilization
            ax1.set_title('GPU Utilization Over Training', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Epochs', fontsize=12)
            ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
            
            if snn_gpu_util and len(snn_gpu_util) > 0:
                snn_plot_data = snn_gpu_util[:max_length] if len(snn_gpu_util) > max_length else snn_gpu_util
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax1.plot(snn_time_steps, snn_plot_data, 'o-', 
                        label='SNN GPU Utilization', linewidth=2, markersize=6, color='lightblue')
            
            if ann_gpu_util and len(ann_gpu_util) > 0:
                ann_plot_data = ann_gpu_util[:max_length] if len(ann_gpu_util) > max_length else ann_gpu_util
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax1.plot(ann_time_steps, ann_plot_data, 's-', 
                        label='ANN GPU Utilization', linewidth=2, markersize=6, color='lightcoral')
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1, max_length)
            ax1.set_ylim(0, max(snn_gpu_util + ann_gpu_util + [100]) * 1.1)
            
            # Chart 2: Memory Utilization
            ax2.set_title('Memory Utilization Over Training', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Training Epochs', fontsize=12)
            ax2.set_ylabel('Memory Utilization (%)', fontsize=12)
            
            if snn_memory_util and len(snn_memory_util) > 0:
                snn_plot_data = snn_memory_util[:max_length] if len(snn_memory_util) > max_length else snn_memory_util
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax2.plot(snn_time_steps, snn_plot_data, 'o-', 
                        label='SNN Memory Utilization', linewidth=2, markersize=6, color='lightblue')
            
            if ann_memory_util and len(ann_memory_util) > 0:
                ann_plot_data = ann_memory_util[:max_length] if len(ann_memory_util) > max_length else ann_memory_util
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax2.plot(ann_time_steps, ann_plot_data, 's-', 
                        label='ANN Memory Utilization', linewidth=2, markersize=6, color='lightcoral')
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(1, max_length)
            ax2.set_ylim(0, max(snn_memory_util + ann_memory_util + [100]) * 1.1)
            
            plt.tight_layout()
            self.save_chart('14_hardware_utilization')
            
        except Exception as e:
            print(f"❌ Error in hardware utilization chart: {e}")
            self.create_fallback_chart('14_hardware_utilization', 'Hardware Utilization Analysis')
    
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
