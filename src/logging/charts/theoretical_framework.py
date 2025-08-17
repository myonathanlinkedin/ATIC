"""
Theoretical Framework Chart Module
Generates theoretical framework validation bar chart for SNN and ANN models.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import sys
import os
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)
try:  # pragma: no cover
    from .base_chart import BaseChart  # type: ignore
except Exception:  # pragma: no cover
    from base_chart import BaseChart  # type: ignore

class TheoreticalFrameworkChart(BaseChart):
    """Theoretical framework validation chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Strictly data-only: read from nested blocks; default to 0.0
            frameworks = [
                'Temporal Binding',
                'Predictive Coding',
                'Neural Synchronization',
            ]

            snn_block = (
                results.get('snn', {}) if isinstance(results, dict) else {}
            )
            ann_block = (
                results.get('ann', {}) if isinstance(results, dict) else {}
            )
            snn_tv = snn_block.get('theoretical_validation', {})
            ann_tv = ann_block.get('theoretical_validation', {})

            snn_scores = [
                float(
                    snn_tv.get('temporal_binding', {}).get(
                        'binding_strength', 0.0
                    )
                ),
                float(
                    snn_tv.get('predictive_coding', {}).get(
                        'prediction_accuracy', 0.0
                    )
                ),
                float(
                    snn_tv.get('neural_synchronization', {}).get(
                        'phase_synchronization', 0.0
                    )
                ),
            ]
            ann_scores = [
                float(
                    ann_tv.get('temporal_binding', {}).get(
                        'binding_strength', 0.0
                    )
                ),
                float(
                    ann_tv.get('predictive_coding', {}).get(
                        'prediction_accuracy', 0.0
                    )
                ),
                float(
                    ann_tv.get('neural_synchronization', {}).get(
                        'phase_synchronization', 0.0
                    )
                ),
            ]
            x = np.arange(len(frameworks))
            width = 0.35
            fig, ax = plt.subplots(
                figsize=(12, 8), constrained_layout=True
            )
            bars1 = ax.bar(x - width/2, snn_scores, width, label='SNN', alpha=0.8)
            bars2 = ax.bar(x + width/2, ann_scores, width, label='ANN', alpha=0.8)
            ax.set_xlabel('Theoretical Framework', fontsize=14)
            ax.set_ylabel('Validation Score', fontsize=14)
            # Reduce main title font size
            fig.suptitle('Theoretical Framework Validation: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.set_xticks(x)
            ax.set_xticklabels(frameworks)
            ax.legend()
            ax.grid(True, alpha=0.3)
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    bar_x = bar.get_x()
                    bar_width = bar.get_width()
                    if isinstance(bar_x, (int, float)) and isinstance(bar_width, (int, float)):
                        text_x = bar_x + bar_width / 2.0
                    else:
                        text_x = 0.5
                    ax.text(text_x, height, f'{height:.3f}', ha='center', va='bottom')
            self.save_chart('29_theoretical_framework')
        except Exception as e:
            print(f"❌ Error in theoretical framework chart: {e}")
            self.create_fallback_chart('29_theoretical_framework', 'Theoretical Framework Validation')
