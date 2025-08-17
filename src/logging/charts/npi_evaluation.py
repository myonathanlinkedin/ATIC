"""
NPI Evaluation Chart Module
Generates Neuromorphic Performance Index evaluation comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
# Use absolute import to work both as module and script
from base_chart import BaseChart


class NPIEvaluationChart(BaseChart):
    """Chart for Neuromorphic Performance Index evaluation"""

    def plot(self, results: Dict[str, Any]) -> None:
        """Generate NPI evaluation chart from raw values in results."""
        try:
            metrics = ['Energy Efficiency', 'Computational Efficiency', 'Memory Efficiency']
            snn_npi = results.get('snn', {}).get('npi_components', [])
            ann_npi = results.get('ann', {}).get('npi_components', [])

            # Debug data extraction
            print(f"🔧 DEBUG: SNN NPI components: {snn_npi}")
            print(f"🔧 DEBUG: ANN NPI components: {ann_npi}")

            # If per-epoch arrays, take last epoch snapshot strictly
            if isinstance(snn_npi, list) and len(snn_npi) > 0 and isinstance(snn_npi[0], list):
                snn_npi = snn_npi[-1]
            if isinstance(ann_npi, list) and len(ann_npi) > 0 and isinstance(ann_npi[0], list):
                ann_npi = ann_npi[-1]

            def norm(arr):
                """Normalize array with real data only - NO FABRICATED VALUES"""
                if not isinstance(arr, list):
                    return [float(arr)] * len(metrics)
                if len(arr) >= len(metrics):
                    return [float(x) for x in arr[:len(metrics)]]
                # NO FABRICATED DATA - use zeros for missing values
                return [float(x) for x in (arr + [0.0] * (len(metrics) - len(arr)))]

            snn_npi = norm(snn_npi)
            ann_npi = norm(ann_npi)

            # Ensure we have valid data
            print(f"🔧 DEBUG: Normalized SNN NPI: {snn_npi}")
            print(f"🔧 DEBUG: Normalized ANN NPI: {ann_npi}")

            # Check if all values are zero (no real data)
            if all(v == 0.0 for v in snn_npi + ann_npi):
                print("⚠️  Warning: All NPI values are zero - no real data available")
                # Show empty chart with clear message - NO FABRICATED DATA
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.text(0.5, 0.5, 'No Valid NPI Data Available\n\n'
                       'All performance metrics returned zero values\n'
                       'Run benchmark with more epochs to generate NPI data',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                                edgecolor='orange', alpha=0.8))
                # Reduce main title font size
                ax.set_title('Neuromorphic Performance Index (NPI): SNN vs ANN\n'
                           'No Valid Data Available', fontsize=14, fontweight='bold')  # Reduced from 16 to 14
                ax.set_xticks([])
                ax.set_yticks([])
                self.save_chart('17_npi_evaluation')
                return

            # Increase figsize for better readability
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for ax, title, sv, av in zip(axes, metrics, snn_npi, ann_npi):
                # Ensure values are always positive for bar height
                eps = 1e-3
                snn_height = max(eps, abs(sv))
                ann_height = max(eps, abs(av))
                
                bars = ax.bar(['SNN', 'ANN'], [snn_height, ann_height],
                              color=['red', 'blue'], alpha=0.85)
                
                # Always display numbers above bars with proper positioning
                for b, v, height in zip(bars, [sv, av], [snn_height, ann_height]):
                    # Calculate text position above bar
                    text_y = v + (v * 0.05)  # 5% above bar
                    
                    # Ensure text is always visible
                    ax.text(
                        b.get_x() + b.get_width()/2.0,
                        text_y,
                        f'{v:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='gray', alpha=0.8)
                    )
                
                # Reduce sub-chart title font sizes
                ax.set_title(title, fontsize=11, fontweight='bold')  # Reduced from 13 to 11
                ax.grid(True, axis='y', alpha=0.3)
                ax.set_ylim(bottom=0)
                
                # Adjust y-axis to accommodate text labels
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.2)
            
            # Reduce main title font size
            fig.suptitle('Neuromorphic Performance Index (NPI): SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            self.save_chart('17_npi_evaluation')
            
        except Exception as e:
            print(f"❌ Error in NPI evaluation chart: {e}")
            self.create_fallback_chart('17_npi_evaluation', 
                                    'Neuromorphic Performance Index Evaluation') 
