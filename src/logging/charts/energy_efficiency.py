"""
Energy Efficiency Chart Module
Generates energy efficiency comparison between SNN and ANN models.
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


class EnergyEfficiencyChart(BaseChart):
    """Energy efficiency comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate energy efficiency comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Extract energy efficiency data
            snn_energy = results.get('snn', {}).get('energy_efficiency', [])
            ann_energy = results.get('ann', {}).get('energy_efficiency', [])
            
            # Prefer JSON series directly; if missing, compute simply; never rescale
            def last_value(val, default):
                if isinstance(val, list) and len(val) > 0:
                    return float(val[-1])
                try:
                    return float(val)
                except Exception:
                    return default
            snn_inference_time = last_value(results.get('snn', {}).get('inference_time', 0.0), 0.0)
            ann_inference_time = last_value(results.get('ann', {}).get('inference_time', 0.0), 0.0)
            snn_energy_per_sample = last_value(results.get('snn', {}).get('energy_per_sample', 0.0), 0.0)
            ann_energy_per_sample = last_value(results.get('ann', {}).get('energy_per_sample', 0.0), 0.0)

            if len(snn_energy) == 0:
                acc = last_value(
                    results.get('snn', {}).get('final_accuracy', 0.0), 0.0
                )
                denom = (snn_inference_time * snn_energy_per_sample)
                snn_energy = [acc / denom] if denom > 0 else [0.0]
            if len(ann_energy) == 0:
                acc = last_value(
                    results.get('ann', {}).get('final_accuracy', 0.0), 0.0
                )
                denom = (ann_inference_time * ann_energy_per_sample)
                ann_energy = [acc / denom] if denom > 0 else [0.0]

            # Align lengths strictly: trim both to the same min length
            if len(snn_energy) == 0 and len(ann_energy) == 0:
                snn_energy = [0.0]
                ann_energy = [0.0]
            elif len(snn_energy) == 0:
                snn_energy = [0.0] * len(ann_energy)
            elif len(ann_energy) == 0:
                ann_energy = [0.0] * len(snn_energy)
            min_length = max(1, min(len(snn_energy), len(ann_energy)))
            snn_energy = snn_energy[:min_length]
            ann_energy = ann_energy[:min_length]
            
            # Create line chart
            fig, ax = plt.subplots(figsize=(12, 8))
            epochs = list(range(1, len(snn_energy) + 1))
            
            ax.plot(epochs, snn_energy, 'o-', label='SNN', linewidth=2, markersize=6, color='red')
            ax.plot(epochs, ann_energy, 's-', label='ANN', linewidth=2, markersize=6, color='blue')
            
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Energy Efficiency (Accuracy / (Time × Energy))', fontsize=14)
            # Reduce main title font size
            fig.suptitle('Energy Efficiency Comparison: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance context
            perf_text = (f'Performance: SNN={snn_inference_time:.3f}s, {snn_energy_per_sample:.6f}J | '
                        f'ANN={ann_inference_time:.3f}s, {ann_energy_per_sample:.6f}J')
            ax.text(0.02, 0.98, perf_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            # Save only as chart 10 (canonical)
            self.save_chart('10_energy_efficiency')
        except Exception as e:
            print(f"❌ Error in energy efficiency chart: {e}")
            # Fallback only for chart 10
            self.create_fallback_chart('10_energy_efficiency', 'Energy Efficiency Comparison')
