"""
Final Accuracy Chart Module
Generates final accuracy comparison between SNN and ANN models.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# Add the charts directory to the path for imports
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import BaseChart directly
from base_chart import BaseChart


class FinalAccuracyChart(BaseChart):
    """Final accuracy comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate final accuracy comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Extract final/test accuracy with robust fallback
            def extract_acc(model_block: Dict[str, Any]) -> float:
                acc = model_block.get('test_accuracy', model_block.get('final_accuracy', 0.0))
                try:
                    acc = float(acc)
                except Exception:
                    acc = 0.0
                # If stored as percent (e.g., 12.4), normalize to [0,1]
                if acc > 1.0:
                    acc = acc / 100.0
                # Clamp to [0,1]
                return max(0.0, min(1.0, acc))

            snn_accuracy = extract_acc(results.get('snn', {}))
            ann_accuracy = extract_acc(results.get('ann', {}))
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            models = ['SNN', 'ANN']
            accuracies = [snn_accuracy, ann_accuracy]
            colors = ['red', 'blue']
            
            bars = ax.bar(models, accuracies, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Model Type', fontsize=14)
            ax.set_ylabel('Final Accuracy', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits (percent scale appearance)
            ymax = max(0.2, max(accuracies) * 1.2)
            ax.set_ylim(0.0, min(1.0, ymax))
            
            # Single title to avoid duplication
            fig.suptitle('Final Accuracy Comparison: SNN vs ANN',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_chart('06_final_accuracy')
        except Exception as e:
            print(f"❌ Error in final accuracy chart: {e}")
            self.create_fallback_chart('06_final_accuracy', 'Final Accuracy Comparison')
