"""
Information Integration Chart Module
Generates information integration over time line chart for SNN and ANN models.
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
class InformationIntegrationChart(BaseChart):
    """Information integration over time chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Extract information integration data with dynamic epoch support
            snn_integration = self.safe_extract_list_data(
                results.get('snn', {}), 'information_integration')
            ann_integration = self.safe_extract_list_data(
                results.get('ann', {}), 'information_integration')
            
            # Auto-detect epoch length from available data
            epoch_length = self._detect_epoch_length(results)
            
            if not snn_integration and not ann_integration:
                print(f"⚠️  Warning: No information integration data found")
                self.create_fallback_chart('27_information_integration', 'Information Integration Analysis')
                return
            
            # Create dynamic time steps based on actual data length
            max_length = max(
                len(snn_integration) if snn_integration else 0,
                len(ann_integration) if ann_integration else 0,
                epoch_length
            )
            
            if max_length == 0:
                self.create_fallback_chart('27_information_integration', 'Information Integration Analysis')
                return
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot SNN information integration
            if snn_integration and len(snn_integration) > 0:
                snn_plot_data = snn_integration[:max_length] if len(snn_integration) > max_length else snn_integration
                snn_time_steps = range(1, len(snn_plot_data) + 1)
                ax.plot(snn_time_steps, snn_plot_data, 'o-', 
                       label='SNN Information Integration', linewidth=2, markersize=6, color='lightblue')
            
            # Plot ANN information integration
            if ann_integration and len(ann_integration) > 0:
                ann_plot_data = ann_integration[:max_length] if len(ann_integration) > max_length else ann_integration
                ann_time_steps = range(1, len(ann_plot_data) + 1)
                ax.plot(ann_time_steps, ann_plot_data, 's-', 
                       label='ANN Information Integration', linewidth=2, markersize=6, color='lightcoral')
            
            ax.set_xlabel('Training Epochs', fontsize=14)
            ax.set_ylabel('Information Integration Index', fontsize=14)
            ax.set_title('Information Integration: SNN vs ANN\nHigher Values Indicate Better Integration', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on actual data
            ax.set_xlim(1, max_length)
            
            plt.tight_layout()
            self.save_chart('27_information_integration')
            
        except Exception as e:
            print(f"❌ Error in information integration chart: {e}")
            self.create_fallback_chart('27_information_integration', 'Information Integration Analysis')
    
    def _detect_epoch_length(self, results: Dict[str, Any]) -> int:
        """Auto-detect epoch length from available data"""
        try:
            # Try to get epoch length from training data
            training_keys = ['training_accuracy', 'training_loss', 'validation_accuracy', 'validation_loss']
            for key in training_keys:
                if key in results:
                    data = results[key]
                    if isinstance(data, list) and len(data) > 0:
                        return len(data)
            
            # Try to get from SNN results
            if 'snn' in results and isinstance(results['snn'], dict):
                for key in training_keys:
                    if key in results['snn']:
                        data = results['snn'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            # Try to get from ANN results
            if 'ann' in results and isinstance(results['ann'], dict):
                for key in training_keys:
                    if key in results['ann']:
                        data = results['ann'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            return 10  # Default fallback
            
        except Exception:
            return 10
