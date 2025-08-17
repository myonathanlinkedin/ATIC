"""
Effect Sizes Chart Module
Generates effect sizes forest plot for SNN and ANN models.
"""
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from base_chart import BaseChart
class EffectSizesChart(BaseChart):
    """Effect sizes forest plot chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            metrics = [
                'Accuracy', 'Energy', 'Memory', 'Speed', 'BPI', 'TEI', 'NPI'
            ]

            snn = results.get('snn', {})
            ann = results.get('ann', {})

            def get_series(block: dict, keys: List[str]) -> List[float]:
                for k in keys:
                    v = block.get(k, []) if isinstance(block, dict) else []
                    if isinstance(v, list) and len(v) > 0:
                        return [float(x) for x in v if np.isfinite(x)]
                return []

            # Gather series from JSON
            snn_acc = get_series(snn, ['validation_accuracy', 'training_accuracy'])
            ann_acc = get_series(ann, ['validation_accuracy', 'training_accuracy'])
            snn_energy = get_series(snn, ['energy_efficiency'])
            ann_energy = get_series(ann, ['energy_efficiency'])
            snn_mem = get_series(snn, ['memory_utilization'])
            ann_mem = get_series(ann, ['memory_utilization'])

            def speeds(block: dict) -> List[float]:
                times = get_series(block, ['inference_times'])
                if len(times) == 0:
                    return []
                result: List[float] = []
                for t in times:
                    t = float(t)
                    if t > 0:
                        result.append(1.0 / t)
                return result

            snn_speed = speeds(snn)
            ann_speed = speeds(ann)

            snn_bpi = get_series(snn, ['bpi_over_time'])
            ann_bpi = get_series(ann, ['bpi_over_time'])

            def flatten_or_empty(v: Any) -> List[float]:
                if isinstance(v, list) and len(v) > 0:
                    return [
                        float(x) for x in v
                        if isinstance(x, (int, float)) and np.isfinite(x)
                    ]
                return []

            snn_tei = flatten_or_empty(snn.get('tei_components', []))
            ann_tei = flatten_or_empty(ann.get('tei_components', []))
            snn_npi = flatten_or_empty(snn.get('npi_components', []))
            ann_npi = flatten_or_empty(ann.get('npi_components', []))

            # Compute Cohen's d and simple 95% CI
            def cohens_d(a: List[float], b: List[float]) -> Tuple[float, float]:
                if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
                    return 0.0, 0.0
                m = min(len(a), len(b))
                a_arr = np.array(a[:m], dtype=float)
                b_arr = np.array(b[:m], dtype=float)
                mask = np.isfinite(a_arr) & np.isfinite(b_arr)
                a_arr, b_arr = a_arr[mask], b_arr[mask]
                if len(a_arr) < 2 or len(b_arr) < 2:
                    return 0.0, 0.0
                n1, n2 = len(a_arr), len(b_arr)
                mean1, mean2 = a_arr.mean(), b_arr.mean()
                var1, var2 = a_arr.var(ddof=1), b_arr.var(ddof=1)
                pooled = np.sqrt(
                    ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)
                )
                d = 0.0 if pooled <= 1e-12 else (mean1 - mean2) / pooled
                se = (
                    np.sqrt(
                        (n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2 - 2))
                    )
                    if (n1 > 1 and n2 > 1)
                    else 0.0
                )
                ci = 1.96 * se
                ci = 0.0 if not np.isfinite(ci) else float(ci)
                return float(d), ci

            series_pairs = [
                (snn_acc, ann_acc),
                (snn_energy, ann_energy),
                (snn_mem, ann_mem),
                (snn_speed, ann_speed),
                (snn_bpi, ann_bpi),
                (snn_tei, ann_tei),
                (snn_npi, ann_npi),
            ]

            effect_sizes: List[float] = []
            confidence_intervals: List[float] = []
            for a, b in series_pairs:
                d, ci = cohens_d(a, b)
                effect_sizes.append(d)
                confidence_intervals.append(ci)

            y_pos = np.arange(len(metrics))
            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
            ax.errorbar(effect_sizes, y_pos, xerr=confidence_intervals, fmt='o', capsize=5)
            ax.axvline(
                x=0,
                color='red',
                linestyle='--',
                alpha=0.7,
                linewidth=2,
                label='No Effect (SNN = ANN)'
            )
            ax.set_xlabel(
                "Effect Size (Cohen's d)\n"
                "Positive: SNN > ANN, Negative: ANN > SNN",
                fontsize=14,
            )
            ax.set_ylabel('Metrics', fontsize=14)
            ax.set_title(
                'Effect Sizes Forest Plot: SNN vs ANN Comparison\n'
                'Positive Values: SNN Better, Negative Values: ANN Better',
                fontsize=14,  # Reduced from 16 to 14
                fontweight='bold',
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            for y, d in zip(y_pos, effect_sizes):
                ax.text(d, y + 0.1, f'{d:.2f}', fontsize=9, ha='center')
            self.save_chart('31_effect_sizes')
        except Exception as e:
            print(f"❌ Error in effect sizes chart: {e}")
            self.create_fallback_chart('31_effect_sizes', 'Effect Sizes Forest Plot')
