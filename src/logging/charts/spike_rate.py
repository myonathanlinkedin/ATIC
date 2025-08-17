"""
Spike Rate Chart Module
Generates spike rate over time heatmap for SNN models.
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
class SpikeRateChart(BaseChart):
    """Spike rate over time heatmap chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Try to get spike rate data from SNN results
            spike_data = None
            
            # Check if results has 'snn' key
            if 'snn' in results and isinstance(results['snn'], dict):
                spike_data = results['snn'].get('spike_rate', None)
            else:
                # Fallback: check if spike_rate is directly in results
                spike_data = results.get('spike_rate', None)
            
            if spike_data is not None and isinstance(spike_data, (list, np.ndarray)) and len(spike_data) > 0:
                # Convert to numpy array if it's a list
                if isinstance(spike_data, list):
                    spike_data = np.array(spike_data)
                
                # Ensure it's a 2D array for heatmap
                if spike_data.ndim == 1:
                    # Reshape 1D array to 2D (assuming square matrix)
                    size = int(np.sqrt(len(spike_data)))
                    if size * size == len(spike_data):
                        spike_data = spike_data.reshape(size, size)
                    else:
                        # Pad to make it square
                        size = int(np.ceil(np.sqrt(len(spike_data))))
                        padded = np.zeros(size * size)
                        padded[:len(spike_data)] = spike_data
                        spike_data = padded.reshape(size, size)
                
                # Ensure minimum size for visualization
                if spike_data.shape[0] < 5 or spike_data.shape[1] < 5:
                    # Pad to minimum size
                    min_size = max(5, spike_data.shape[0], spike_data.shape[1])
                    padded = np.zeros((min_size, min_size))
                    padded[:spike_data.shape[0], :spike_data.shape[1]] = spike_data
                    spike_data = padded
                
                fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
                im = ax.imshow(spike_data, cmap='hot', aspect='auto')
                ax.set_xlabel('Neurons', fontsize=14)
                ax.set_ylabel('Time Steps', fontsize=14)
                # Remove redundant main title since subtitle is clear
                ax.set_title('Spike Rate Over Time Heatmap: SNN\nHigher Spike Rates Indicate More Active Neural Processing', fontsize=14, fontweight='bold')  # Reduced from 16 to 14
                plt.colorbar(im, ax=ax, shrink=0.8)
                self.save_chart('20_spike_rate')
            else:
                print(f"⚠️  Warning: No valid spike rate data found. Data type: {type(spike_data)}")
                self.create_fallback_chart('20_spike_rate', 'Spike Rate Over Time')
        except Exception as e:
            print(f"❌ Error in spike rate chart: {e}")
            self.create_fallback_chart('20_spike_rate', 'Spike Rate Over Time')
