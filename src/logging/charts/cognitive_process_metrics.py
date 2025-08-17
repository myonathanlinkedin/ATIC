"""
Cognitive Process Metrics Chart Module
Generates cognitive process metrics comparison between SNN and ANN models.
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
class CognitiveProcessMetricsChart(BaseChart):
    """Cognitive process metrics chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            snn_cognitive = results.get('snn', {}).get('cognitive_processes', None)
            ann_cognitive = results.get('ann', {}).get('cognitive_processes', None)
            if snn_cognitive and ann_cognitive:
                attention_metrics = ['focus_score', 'selectivity', 'sustained_attention']
                memory_metrics = ['working_memory', 'episodic_memory', 'semantic_memory']
                executive_metrics = ['planning', 'decision_making', 'cognitive_flexibility']
                snn_attention = [snn_cognitive.get(m, 0.0) for m in attention_metrics]
                snn_memory = [snn_cognitive.get(m, 0.0) for m in memory_metrics]
                snn_executive = [snn_cognitive.get(m, 0.0) for m in executive_metrics]
                ann_attention = [ann_cognitive.get(m, 0.0) for m in attention_metrics]
                ann_memory = [ann_cognitive.get(m, 0.0) for m in memory_metrics]
                ann_executive = [ann_cognitive.get(m, 0.0) for m in executive_metrics]
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                width = 0.35

                def draw_group(ax, labels_list, snn_vals, ann_vals):
                    x = np.arange(len(labels_list))
                    eps = 0.005  # minimal visible height for zero values
                    snn_disp = [v if v > 0 else eps for v in snn_vals]
                    ann_disp = [v if v > 0 else eps for v in ann_vals]
                    bars1 = ax.bar(x - width/2, snn_disp, width, label='SNN', alpha=0.8, color='lightcoral')
                    bars2 = ax.bar(x + width/2, ann_disp, width, label='ANN', alpha=0.8, color='lightblue')
                    # annotate actual values including zeros
                    for b, v in zip(bars1, snn_vals):
                        ax.text(b.get_x() + b.get_width()/2., max(b.get_height(), eps) + 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                    for b, v in zip(bars2, ann_vals):
                        ax.text(b.get_x() + b.get_width()/2., max(b.get_height(), eps) + 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels_list, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_ylim(0, 1.05)

                x1 = np.arange(len(attention_metrics))
                draw_group(ax1, attention_metrics, snn_attention, ann_attention)
                ax1.set_title('Attention Analysis', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Metric Value', fontsize=10)
                ax1.legend(fontsize=9)

                x2 = np.arange(len(memory_metrics))
                draw_group(ax2, memory_metrics, snn_memory, ann_memory)
                ax2.set_title('Memory Analysis', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Metric Value', fontsize=10)
                ax2.legend(fontsize=9)

                x3 = np.arange(len(executive_metrics))
                draw_group(ax3, executive_metrics, snn_executive, ann_executive)
                ax3.set_title('Executive Analysis', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Metric Value', fontsize=10)
                ax3.legend(fontsize=9)
                fig.suptitle('Cognitive Process Metrics: SNN vs ANN',
                        fontsize=14, fontweight='bold')
                self.save_chart('19_cognitive_process_metrics')
            else:
                self.create_fallback_chart('19_cognitive_process_metrics', 'Cognitive Process Metrics')
        except Exception as e:
            print(f"❌ Error in cognitive process metrics chart: {e}")
            self.create_fallback_chart('19_cognitive_process_metrics', 'Cognitive Process Metrics')
