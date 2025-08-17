"""
Statistical Tests Chart Module
Generates statistical tests p-value heatmap for SNN and ANN models.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Dict, Any
import sys
import os
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)
from base_chart import BaseChart
class StatisticalTestsChart(BaseChart):
    """Statistical tests p-value heatmap chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            metrics = ['SNN_Accuracy', 'ANN_Accuracy', 'SNN_Energy', 'ANN_Energy', 'SNN_Memory', 'ANN_Memory', 'SNN_Speed', 'ANN_Speed']

            def get_series(block: dict, keys: list[str]) -> list[float]:
                for k in keys:
                    v = block.get(k, []) if isinstance(block, dict) else []
                    if isinstance(v, list) and len(v) > 1:
                        return [float(x) for x in v if np.isfinite(x)]
                return []

            snn = results.get('snn', {})
            ann = results.get('ann', {})

            snn_acc = get_series(snn, ['validation_accuracy', 'training_accuracy'])
            ann_acc = get_series(ann, ['validation_accuracy', 'training_accuracy'])
            snn_energy = get_series(snn, ['energy_efficiency'])
            ann_energy = get_series(ann, ['energy_efficiency'])
            snn_mem = get_series(snn, ['memory_utilization'])
            ann_mem = get_series(ann, ['memory_utilization'])

            def speeds(block):
                times = get_series(block, ['inference_times'])
                return [float(1.0/max(t, 1e-9)) for t in times] if times else []

            snn_speed = speeds(snn)
            ann_speed = speeds(ann)

            series = [snn_acc, ann_acc, snn_energy, ann_energy, snn_mem, ann_mem, snn_speed, ann_speed]

            # Simple p-value proxy using normal approximation
            def p_value(a: list[float], b: list[float]) -> float:
                if len(a) < 2 or len(b) < 2:
                    return float('nan')
                m = min(len(a), len(b))
                a_arr = np.array(a[:m], dtype=float)
                b_arr = np.array(b[:m], dtype=float)
                mask = np.isfinite(a_arr) & np.isfinite(b_arr)
                if mask.sum() < 2:
                    return float('nan')
                a_arr = a_arr[mask]
                b_arr = b_arr[mask]
                diff = abs(a_arr.mean() - b_arr.mean())
                denom = np.sqrt(a_arr.var(ddof=1)/len(a_arr) + b_arr.var(ddof=1)/len(b_arr)) + 1e-9
                z = diff / denom
                return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z/np.sqrt(2)))))

            n = len(metrics)
            p_values = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        p_values[i, j] = 0.0
                    else:
                        p_values[i, j] = p_value(series[i], series[j])

            fig, ax = plt.subplots(figsize=(10, 8))
            # Replace NaNs with 1.0 (no significance) to avoid rendering gaps
            p_values = np.where(np.isfinite(p_values), p_values, 1.0)
            im = ax.imshow(p_values, cmap='Reds', aspect='auto', vmin=0.0, vmax=1.0)
            for i in range(n):
                for j in range(n):
                    val = p_values[i, j]
                    txt = f'{val:.3f}'
                    ax.text(j, i, txt, ha="center", va="center", color="black")
            ax.set_xlabel('Metrics (SNN vs ANN)', fontsize=14)
            ax.set_ylabel('Metrics (SNN vs ANN)', fontsize=14)
            ax.set_title('Statistical Tests P-Value Heatmap: SNN vs ANN\nLower P-Values Indicate Significant Differences', fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.set_xticks(range(n))
            ax.set_xticklabels(metrics, rotation=45)
            ax.set_yticks(range(n))
            ax.set_yticklabels(metrics)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.grid(False)
            self.save_chart('32_statistical_tests')
        except Exception as e:
            print(f"❌ Error in statistical tests chart: {e}")
            self.create_fallback_chart('32_statistical_tests', 'Statistical Tests P-Value Heatmap')
