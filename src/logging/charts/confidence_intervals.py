"""
Confidence Intervals Chart Module
Generates confidence intervals error bar chart for SNN and ANN models.
"""
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from base_chart import BaseChart
class ConfidenceIntervalsChart(BaseChart):
    """Confidence intervals error bar chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            metrics = ['Accuracy', 'Energy', 'Memory', 'Speed']

            def series(block: dict, keys: List[str]) -> List[float]:
                for k in keys:
                    v = block.get(k, []) if isinstance(block, dict) else []
                    if isinstance(v, list) and len(v) > 0:
                        return [float(x) for x in v if np.isfinite(x)]
                return []

            def speed_series(block: dict) -> List[float]:
                # Use more robust speed calculation to prevent extreme values
                times = series(block, ['inference_times', 'inference_time'])
                if not times:
                    return []
                
                # Filter out unrealistic inference times
                # Too fast (< 0.01s) suggests measurement error, too slow (> 10s) suggests system issue
                valid_times = [t for t in times if 0.01 <= t <= 10.0]
                if not valid_times:
                    return []
                
                # Use log-based speed calculation to prevent extreme values
                # Speed = log(1 + 1/t) instead of 1/t to compress the range
                speeds = [np.log(1 + 1.0/t) for t in valid_times]
                
                # Additional outlier removal for speed values
                if len(speeds) > 2:
                    speeds_array = np.array(speeds)
                    q25, q75 = np.percentile(speeds_array, [25, 75])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    speeds = [s for s in speeds if lower_bound <= s <= upper_bound]
                
                return speeds

            def mean_ci95(values: List[float]) -> Tuple[float, float]:
                if not values or len(values) < 2:
                    return 0.0, 0.0
                arr = np.array(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) < 2:
                    return 0.0, 0.0
                m = float(arr.mean())
                se = float(arr.std(ddof=1)) / np.sqrt(len(arr))
                return m, 1.96 * se

            snn_blk = results.get('snn', {})
            ann_blk = results.get('ann', {})

            snn_means: List[float] = []
            snn_errs: List[float] = []
            ann_means: List[float] = []
            ann_errs: List[float] = []

            # Accuracy - Use correct data keys
            m, e = mean_ci95(series(snn_blk, ['train_accuracies', 'val_accuracies', 'training_accuracy', 'validation_accuracy']))
            snn_means.append(m)
            snn_errs.append(e)
            m, e = mean_ci95(series(ann_blk, ['train_accuracies', 'val_accuracies', 'training_accuracy', 'validation_accuracy']))
            ann_means.append(m)
            ann_errs.append(e)
            
            # Energy efficiency - Use correct data keys
            m, e = mean_ci95(series(snn_blk, ['energy_efficiency', 'energy_per_sample']))
            snn_means.append(m)
            snn_errs.append(e)
            m, e = mean_ci95(series(ann_blk, ['energy_efficiency', 'energy_per_sample']))
            ann_means.append(m)
            ann_errs.append(e)
            
            # Memory utilization - Use correct data keys
            m, e = mean_ci95(series(snn_blk, ['memory_utilization', 'memory_usage']))
            snn_means.append(m)
            snn_errs.append(e)
            m, e = mean_ci95(series(ann_blk, ['memory_utilization', 'memory_usage']))
            ann_means.append(m)
            ann_errs.append(e)
            # Speed (samples/sec)
            m, e = mean_ci95(speed_series(snn_blk))
            snn_means.append(m)
            snn_errs.append(e)
            m, e = mean_ci95(speed_series(ann_blk))
            ann_means.append(m)
            ann_errs.append(e)

            x = np.arange(len(metrics))
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
            ax.errorbar(x - width/2, snn_means, yerr=snn_errs, fmt='o', label='SNN', capsize=5, capthick=2)
            ax.errorbar(x + width/2, ann_means, yerr=ann_errs, fmt='s', label='ANN', capsize=5, capthick=2)
            ax.set_xlabel('Metrics', fontsize=14)
            ax.set_ylabel('Values with 95% CI', fontsize=14)
            # Reduce main title font size
            fig.suptitle('Confidence Intervals Analysis: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.save_chart('30_confidence_intervals')
        except Exception as e:
            print(f"❌ Error in confidence intervals chart: {e}")
            self.create_fallback_chart('30_confidence_intervals', 'Confidence Intervals Analysis')
