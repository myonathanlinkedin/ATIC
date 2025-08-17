"""
Model Metrics Chart Module
Generates comprehensive model metrics comparison between SNN and ANN models.
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


class ModelMetricsChart(BaseChart):
    """Model metrics comparison chart"""
    
    def plot(self, results: Dict[str, Any]) -> None:
        """
        Generate model metrics comparison chart
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        try:
            # Extract data from correct locations
            snn_data = results.get('snn', {})
            ann_data = results.get('ann', {})
            
            # Safe extraction with multiple fallback keys
            def safe_extract_metric(data_dict, key, fallback_keys=None):
                """Extract metric with multiple fallback options"""
                if fallback_keys is None:
                    fallback_keys = []
                
                # Try primary key first
                value = data_dict.get(key, None)
                if value is not None:
                    try:
                        # Handle list values (e.g., energy_per_sample is a list)
                        if isinstance(value, list) and len(value) > 0:
                            # Take the last value from the list (most recent)
                            return float(value[-1])
                        else:
                            return float(value)
                    except (ValueError, TypeError):
                        pass
                
                # Try fallback keys
                for fallback_key in fallback_keys:
                    value = data_dict.get(fallback_key, None)
                    if value is not None:
                        try:
                            # Handle list values for fallback keys too
                            if isinstance(value, list) and len(value) > 0:
                                return float(value[-1])
                            else:
                                return float(value)
                        except (ValueError, TypeError):
                            continue
                
                # Try metrics sub-dictionary
                metrics = data_dict.get('metrics', {})
                if isinstance(metrics, dict):
                    value = metrics.get(key, None)
                    if value is not None:
                        try:
                            # Handle list values in metrics sub-dict
                            if isinstance(value, list) and len(value) > 0:
                                return float(value[-1])
                            else:
                                return float(value)
                        except (ValueError, TypeError):
                            pass
                    
                    # Try fallback keys in metrics
                    for fallback_key in fallback_keys:
                        value = metrics.get(fallback_key, None)
                        if value is not None:
                            try:
                                # Handle list values for fallback keys in metrics
                                if isinstance(value, list) and len(value) > 0:
                                    return float(value[-1])
                                else:
                                    return float(value)
                            except (ValueError, TypeError):
                                continue
                
                return 0.0  # Final fallback
            
            # Use correct energy data keys
            snn_values = [
                safe_extract_metric(snn_data, 'parameter_count', 
                                  ['total_parameters']) / 1_000_000.0,
                safe_extract_metric(snn_data, 'memory_usage', 
                                  ['memory_utilization', 'mem_used']),
                safe_extract_metric(snn_data, 'energy_per_sample', 
                                  ['energy_consumption', 'energy_total']),
            ]
            ann_values = [
                safe_extract_metric(ann_data, 'parameter_count', 
                                  ['total_parameters']) / 1_000_000.0,
                safe_extract_metric(ann_data, 'memory_usage', 
                                  ['memory_utilization', 'mem_used']),
                safe_extract_metric(ann_data, 'energy_per_sample', 
                                  ['energy_consumption', 'energy_total']),
            ]
            
            # Validate energy values and handle zero values gracefully
            if snn_values[2] <= 0.0 or ann_values[2] <= 0.0:
                print("⚠️  Warning: Energy values are zero - this may indicate measurement issues")
                # Don't use fallback estimates - show actual zero values
                # This ensures transparency about data quality
            
            # Create 1x3 subplots to avoid scale masking and make each metric readable
            fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

            plots = [
                ('Parameter Count (M)', [snn_values[0], ann_values[0]]),
                ('Memory Usage (MB)', [snn_values[1], ann_values[1]]),
                ('Energy Consumption (W)', [snn_values[2], ann_values[2]]),
            ]

            # Reduce main title font size
            fig.suptitle('Model Metrics Comparison: SNN vs ANN',
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            
            for ax, (title, vals) in zip(axes, plots):
                # Ensure values are always positive for bar height
                eps = 1e-3
                snn_height = max(eps, abs(vals[0]))
                ann_height = max(eps, abs(vals[1]))
                
                bars = ax.bar(['SNN', 'ANN'], [snn_height, ann_height],
                              color=['red', 'blue'], alpha=0.85)
                
                # Always display numbers above bars with proper positioning
                for b, v, height in zip(bars, vals, [snn_height, ann_height]):
                    # Calculate text position above bar
                    text_y = height + (height * 0.05)  # 5% above bar
                    
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
            
            self.save_chart('11_model_metrics')
            
        except Exception as e:
            print(f"❌ Error in model metrics chart: {e}")
            self.create_fallback_chart('11_model_metrics', 
                                    'Model Metrics Comparison')
