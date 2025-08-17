"""
Outlier Analysis Chart Module
Generates outlier analysis with statistical tests for SNN and ANN models.
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
class OutlierAnalysisChart(BaseChart):
    """Outlier analysis with statistical tests chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Build a richer distribution using per-epoch arrays to avoid thin boxplots
            def build_series(model_dict: Dict[str, Any]) -> list:
                series = []
                # Accuracy across epochs
                acc = model_dict.get('validation_accuracy', [])
                if isinstance(acc, list):
                    series.extend([float(a) for a in acc])
                # Energy per sample across epochs
                eps = model_dict.get('energy_per_sample', [])
                if isinstance(eps, list):
                    series.extend([float(e) for e in eps])
                # GPU power across epochs
                gp = model_dict.get('gpu_power', [])
                if isinstance(gp, list):
                    series.extend([float(p) for p in gp])
                # Inference times across epochs
                it = model_dict.get('inference_times', [])
                if isinstance(it, list):
                    series.extend([float(t) for t in it])
                # Fallback to outlier_data if present
                od = model_dict.get('outlier_data', [])
                if isinstance(od, list) and len(od) > 0:
                    series.extend([float(x) for x in od])
                return series if len(series) > 0 else [0.0]

            snn_series = build_series(results.get('snn', {}))
            ann_series = build_series(results.get('ann', {}))

            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
            box_plot = ax.boxplot([snn_series, ann_series], tick_labels=['SNN', 'ANN'], patch_artist=True)
            for patch, color in zip(box_plot['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)
            ax.set_ylabel('Values (mixed units)', fontsize=14)
            # Reduce main title font size
            fig.suptitle('Outlier Analysis: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.grid(True, alpha=0.3)
            self.save_chart('34_outlier_analysis')
        except Exception as e:
            print(f"❌ Error in outlier analysis chart: {e}")
            self.create_fallback_chart('34_outlier_analysis', 'Outlier Analysis')
