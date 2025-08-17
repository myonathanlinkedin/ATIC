"""
Spike Timing Chart Module
Generates spike timing over time line chart for SNN models only.
ANN models don't have spike timing, so only SNN data is displayed.
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

try:  # pragma: no cover
    from .base_chart import BaseChart  # type: ignore
except Exception:  # pragma: no cover
    from base_chart import BaseChart  # type: ignore


class SpikeTimingChart(BaseChart):
    """Spike timing over time chart - SNN ONLY"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Only SNN has spike timing, ANN doesn't!
            snn_timing = results.get('snn', {}).get('spike_timing', [])
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_timing:
                print("⚠️  Warning: No SNN spike timing data found")
                self.create_fallback_chart('24_spike_timing',
                                        'SNN Spike Timing Over Time')
                return
            
            # Handle per-epoch data (list of lists)
            if isinstance(snn_timing[0], list):
                # Multi-epoch data: plot each epoch as separate line
                self._plot_multi_epoch_timing(snn_timing, epoch_length)
            else:
                # Single epoch data: plot as single line
                self._plot_single_epoch_timing(snn_timing, epoch_length)
                
        except Exception as e:
            print(f"❌ Error in spike timing chart: {e}")
            self.create_fallback_chart('24_spike_timing',
                                    'SNN Spike Timing Over Time')
    
    def _plot_multi_epoch_timing(self, snn_timing: list, epoch_length: int):
        """Plot spike timing for multiple epochs - SNN ONLY"""
        try:
            # Show ALL epochs, not just 10!
            total_epochs = len(snn_timing)
            
            # Single chart for SNN only, no ANN
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))  # Increased size for more epochs
            
            # Chart: SNN Spike Timing per Epoch
            ax.set_title('SNN Spike Timing: Per-Epoch Progression',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Spike Timing', fontsize=12)
            
            # Generate DISTINCT colors for ALL epochs with high contrast
            # Use a combination of different color maps for better distinction
            if total_epochs <= 6:
                # For small number of epochs, use distinct colors
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            elif total_epochs <= 12:
                # For medium number, combine different color maps
                colors1 = plt.cm.Set1(np.linspace(0, 1, min(8, total_epochs)))
                colors2 = plt.cm.Set2(np.linspace(0, 1, min(8, total_epochs)))
                colors = list(colors1) + list(colors2)
                colors = colors[:total_epochs]
            else:
                # For many epochs, use tab20 for maximum distinction
                colors = plt.cm.tab20(np.linspace(0, 1, total_epochs))
            
            valid_epochs = 0
            
            # Plot ALL epochs with distinct colors
            for i, epoch_data in enumerate(snn_timing):
                if isinstance(epoch_data, list) and len(epoch_data) > 0:
                    # Validate data is not all zeros
                    if any(val != 0.0 for val in epoch_data):
                        time_steps = range(1, len(epoch_data) + 1)
                        ax.plot(time_steps, epoch_data,
                               color=colors[i], alpha=0.9, linewidth=2.5,  # Increased alpha and linewidth
                               label=f'Epoch {i+1}',  # Show ALL epoch labels
                               marker='o', markersize=4)  # Increased marker size
                        valid_epochs += 1
                    else:
                        print(f"⚠️  Warning: Epoch {i+1} has all zero spike timing data")
            
            if valid_epochs == 0:
                ax.text(0.5, 0.5,
                       'No valid spike timing data found\nAll epochs show zero values',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                                alpha=0.8))
                print("⚠️  Warning: All epochs have zero spike timing - possible data collection issue")
            else:
                # Update info box to show total epochs
                ax.text(0.02, 0.98,
                       f'Total epochs: {total_epochs}\nValid epochs: {valid_epochs}',
                       transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                                alpha=0.7))
                
                # Move legend to bottom left and improve layout
                # Use multiple columns and better positioning to avoid covering chart data
                if total_epochs <= 10:
                    # For few epochs, use 2 columns
                    ax.legend(loc='lower left', fontsize=10, ncol=2, 
                             bbox_to_anchor=(0.02, 0.02), framealpha=0.9)
                elif total_epochs <= 20:
                    # For medium epochs, use 3 columns
                    ax.legend(loc='lower left', fontsize=9, ncol=3, 
                             bbox_to_anchor=(0.02, 0.02), framealpha=0.9)
                else:
                    # For many epochs, use 4 columns and smaller font
                    ax.legend(loc='lower left', fontsize=8, ncol=4, 
                             bbox_to_anchor=(0.02, 0.02), framealpha=0.9)
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.save_chart('24_spike_timing')
            
        except Exception as e:
            print(f"❌ Error in multi-epoch spike timing chart: {e}")
            self.create_fallback_chart('24_spike_timing',
                                    'SNN Spike Timing Over Time')
    
    def _plot_single_epoch_timing(self, snn_timing: list, epoch_length: int):
        """Plot spike timing for single epoch - SNN ONLY"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax.set_title('SNN Spike Timing: Single Epoch',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Spike Timing', fontsize=12)
            
            if snn_timing and len(snn_timing) > 0:
                # Validate data is not all zeros
                if any(val != 0.0 for val in snn_timing):
                    time_steps = range(1, len(snn_timing) + 1)
                    ax.plot(time_steps, snn_timing,
                           color='blue', alpha=0.8, linewidth=2,
                           marker='o', markersize=4)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5,
                           'No valid spike timing data found\nAll values are zero',
                           transform=ax.transAxes, ha='center', va='center', fontsize=14,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                                    alpha=0.8))
                    print("⚠️  Warning: Single epoch has all zero spike timing - possible data collection issue")
            else:
                ax.text(0.5, 0.5, 'No spike timing data available',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
            
            plt.tight_layout()
            self.save_chart('24_spike_timing')
            
        except Exception as e:
            print(f"❌ Error in single epoch spike timing chart: {e}")
            self.create_fallback_chart('24_spike_timing',
                                    'SNN Spike Timing Over Time')
    
    def _detect_epoch_length(self, results: Dict[str, Any]) -> int:
        """Detect epoch length from results data"""
        try:
            # Try to get epoch length from training data
            snn_data = results.get('snn', {})
            if 'train_accuracies' in snn_data and snn_data['train_accuracies']:
                return len(snn_data['train_accuracies'])
            elif 'training_accuracy' in snn_data:
                return 1
            else:
                return 20  # Default fallback
        except Exception:
            return 20  # Default fallback
