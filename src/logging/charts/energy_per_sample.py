"""
Energy Per Sample Chart Module
Generates energy per sample distribution comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
from typing import Dict, Any
import sys
import os
import scipy.stats as stats  # noqa: F401

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import BaseChart with package-safe fallback
try:  # pragma: no cover
    from .base_chart import BaseChart  # type: ignore
except Exception:  # pragma: no cover
    from base_chart import BaseChart  # type: ignore


class EnergyPerSampleChart(BaseChart):
    """Energy per sample distribution comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate energy per sample comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Explicit matplotlib configuration to prevent external annotations
            plt.rcParams['figure.autolayout'] = False
            plt.rcParams['figure.constrained_layout.use'] = False
            
            # Extract energy per sample; JSON may contain per-epoch lists
            snn_energy = results.get('snn', {}).get('energy_per_sample', None)
            ann_energy = results.get('ann', {}).get('energy_per_sample', None)

            # If lists are provided, use last; if missing, use 0.0 (neutral)
            if isinstance(snn_energy, list):
                snn_energy = snn_energy[-1] if snn_energy else 0.0
            if isinstance(ann_energy, list):
                ann_energy = ann_energy[-1] if ann_energy else 0.0

            snn_energy = float(snn_energy) if snn_energy is not None else 0.0
            ann_energy = float(ann_energy) if ann_energy is not None else 0.0

            if snn_energy == 0.0 and ann_energy == 0.0:
                self.create_fallback_chart('08_energy_per_sample', 'Energy per Sample Comparison')
                return
            
            # Create chart with explicit boundaries and no external annotations
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Clear any existing annotations or text
            ax.clear()
            
            models = ['SNN', 'ANN']
            energy_values = [snn_energy, ann_energy]
            colors = ['red', 'blue']
            
            # Generate bars with explicit positioning
            bars = ax.bar(models, energy_values, color=colors, alpha=0.8, width=0.6)
            
            # Add value labels on bars with explicit positioning
            for bar, energy_val in zip(bars, energy_values):
                height = bar.get_height()
                # Position text above bar with explicit coordinates
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + (height * 0.05),  # 5% above bar
                    f'{energy_val:.3f}J', 
                    ha='center', 
                    va='bottom', 
                    fontsize=11, 
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.9
                    )
                )
            
            # Set explicit chart boundaries
            ax.set_xlabel('Model Type', fontsize=12)
            ax.set_ylabel('Energy per Sample (Joules)', fontsize=12)
            
            # Ensure text box matches visual bars exactly
            snn_display = snn_energy if snn_energy > 0.0 else 0.0
            ann_display = ann_energy if ann_energy > 0.0 else 0.0
            difference = abs(snn_display - ann_display)
            
            # Calculate subtitle for energy efficiency comparison
            if snn_display > 0 and ann_display > 0:
                if snn_display < ann_display:
                    subtitle = f'SNN is {((ann_display - snn_display) / ann_display * 100):.1f}% more energy efficient'
                else:
                    subtitle = f'ANN is {((snn_display - ann_display) / snn_display * 100):.1f}% more energy efficient'
            else:
                subtitle = 'Energy efficiency comparison'
            
            # Set title with reduced font size
            ax.set_title('Energy per Sample Comparison: SNN vs ANN\n'
                        f'{subtitle}\n'
                        'Lower values indicate better energy efficiency', 
                        fontsize=14, fontweight='bold')
            
            # Add grid with explicit styling
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            
            # Set explicit y-axis limits to prevent external scaling
            max_val = max(energy_values) if max(energy_values) > 0 else 0.1
            ax.set_ylim(0, max_val * 1.3)
            
            # Set explicit x-axis limits
            ax.set_xlim(-0.5, len(models) - 0.5)
            
            # Add performance context with explicit positioning
            perf_text = (
                f'SNN: {snn_display:.3f}J\n'
                f'ANN: {ann_display:.3f}J\n'
                f'Diff: {difference:.3f}J'
            )
            
            # Position text box at bottom-right with explicit coordinates
            ax.text(
                0.98,
                0.02,
                perf_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.9
                ),
                fontweight='bold'
            )
            
            # Use tight_layout with explicit padding
            plt.tight_layout(pad=2.0)
            
            # Ensure no external annotations can be added
            fig.canvas.draw()
            
            # Standardized filename numbering to keep chart count consistent
            self.save_chart('08_energy_per_sample')
            
            # Close figure to prevent memory leaks
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Error in energy per sample chart: {e}")
            self.create_fallback_chart('08_energy_per_sample', 'Energy per Sample Comparison')
