"""
Correlation Matrix Chart Module
Generates correlation matrix heatmap for SNN and ANN models.
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


class CorrelationMatrixChart(BaseChart):
    """Correlation matrix heatmap chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            metrics = ['SNN_Accuracy', 'ANN_Accuracy', 'SNN_Energy', 'ANN_Energy', 
                      'SNN_Memory', 'ANN_Memory', 'SNN_Speed', 'ANN_Speed']

            def get_series(block: dict, keys: list[str]) -> list[float]:
                """Return first available key series as list of floats"""
                for k in keys:
                    v = block.get(k, []) if isinstance(block, dict) else []
                    if isinstance(v, list) and len(v) > 0:
                        # Filter out invalid values
                        valid_values = [float(x) for x in v if np.isfinite(x) and x != 0.0]
                        if len(valid_values) >= 3:  # Need at least 3 values for correlation
                            return valid_values
                return []

            snn = results.get('snn', {})
            ann = results.get('ann', {})

            # Use correct data keys
            # Accuracy - try multiple keys
            snn_acc = get_series(snn, ['val_accuracies', 'train_accuracies', 
                                       'validation_accuracy', 'training_accuracy'])
            ann_acc = get_series(ann, ['val_accuracies', 'train_accuracies', 
                                       'validation_accuracy', 'training_accuracy'])
            
            # Energy efficiency - try multiple keys
            snn_energy = get_series(snn, ['energy_per_sample', 'energy_efficiency', 
                                         'energy_per_sample'])
            ann_energy = get_series(ann, ['energy_per_sample', 'energy_efficiency', 
                                         'energy_per_sample'])
            
            # Memory utilization - try multiple keys
            snn_mem = get_series(snn, ['memory_usage', 'memory_utilization', 
                                      'memory_usage'])
            ann_mem = get_series(ann, ['memory_usage', 'memory_utilization', 
                                      'memory_usage'])
            
            # Speed from inference times - try multiple keys
            def speeds(block):
                times = get_series(block, ['inference_times', 'inference_time'])
                if len(times) == 0:
                    return []
                # Calculate speed properly
                return [float(1.0/max(t, 1e-6)) for t in times if t > 0]
            
            snn_speed = speeds(snn)
            ann_speed = speeds(ann)

            # Debug data extraction
            print(f"🔧 DEBUG: SNN Accuracy: {len(snn_acc)} values")
            print(f"🔧 DEBUG: ANN Accuracy: {len(ann_acc)} values")
            print(f"🔧 DEBUG: SNN Energy: {len(snn_energy)} values")
            print(f"🔧 DEBUG: ANN Energy: {len(ann_energy)} values")
            print(f"🔧 DEBUG: SNN Memory: {len(snn_mem)} values")
            print(f"🔧 DEBUG: ANN Memory: {len(ann_mem)} values")
            print(f"🔧 DEBUG: SNN Speed: {len(snn_speed)} values")
            print(f"🔧 DEBUG: ANN Speed: {len(ann_speed)} values")

            # Build raw series list
            raw_series = [snn_acc, ann_acc, snn_energy, ann_energy, 
                         snn_mem, ann_mem, snn_speed, ann_speed]
            
            n = len(metrics)
            correlation_matrix = np.full((n, n), np.nan)
            
            # Pairwise correlation with better validation
            for i in range(n):
                for j in range(n):
                    a = raw_series[i] if i < len(raw_series) else []
                    b = raw_series[j] if j < len(raw_series) else []
                    
                    # Need at least 3 valid values for correlation
                    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3:
                        correlation_matrix[i, j] = np.nan
                        continue
                    
                    # Align series lengths
                    m = min(len(a), len(b))
                    a_arr = np.array(a[:m], dtype=float)
                    b_arr = np.array(b[:m], dtype=float)
                    
                    # Filter out invalid values
                    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (a_arr != 0) & (b_arr != 0)
                    if mask.sum() < 3:
                        correlation_matrix[i, j] = np.nan
                        continue
                    
                    # Calculate correlation
                    with np.errstate(invalid='ignore'):
                        corr = np.corrcoef(a_arr[mask], b_arr[mask])
                    correlation_matrix[i, j] = corr[0, 1] if corr.shape == (2, 2) else np.nan

            # Check if we have any valid correlations
            valid_correlations = np.sum(np.isfinite(correlation_matrix))
            print(f"🔧 DEBUG: Valid correlations: {valid_correlations}/{n*n}")
            
            if valid_correlations == 0:
                print("⚠️  Warning: No valid correlations found - chart will be empty")
                # NO FABRICATED DATA - show empty chart with clear message
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.text(0.5, 0.5, 'No Valid Correlation Data Available\n\n'
                       'All metrics returned zero or insufficient data\n'
                       'Run benchmark with more epochs to generate correlations',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                                edgecolor='orange', alpha=0.8))
                ax.set_xlabel('Metrics (SNN vs ANN)', fontsize=14)
                ax.set_ylabel('Metrics (SNN vs ANN)', fontsize=14)
                ax.set_title('Correlation Matrix Analysis: SNN vs ANN\n'
                           'No Valid Data Available', fontsize=14, fontweight='bold')  # Reduced from 16 to 14
                ax.set_xticks([])
                ax.set_yticks([])
                self.save_chart('33_correlation_matrix')
                return

            # Make chart wider for better readability
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Display correlation values with better formatting
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    val = correlation_matrix[i, j]
                    if np.isfinite(val):
                        txt = f'{val:.2f}'
                        color = "white" if abs(val) > 0.7 else "black"
                    else:
                        txt = 'NaN'
                        color = "black"
                    ax.text(j, i, txt, ha="center", va="center", color=color, 
                           fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Metrics (SNN vs ANN)', fontsize=14)
            ax.set_ylabel('Metrics (SNN vs ANN)', fontsize=14)
            ax.set_title('Correlation Matrix Analysis: SNN vs ANN\nPositive: Similar Behavior, Negative: Different Behavior', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.grid(False)
            
            self.save_chart('33_correlation_matrix')
            
        except Exception as e:
            print(f"❌ Error in correlation matrix chart: {e}")
            self.create_fallback_chart('33_correlation_matrix', 'Correlation Matrix Analysis')
