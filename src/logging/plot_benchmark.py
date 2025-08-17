"""
Benchmark Plot Dispatcher
Coordinates the generation of 34 different benchmark charts.
Acts as a dispatcher that imports and calls individual chart modules.
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any
import json

# Add the charts directory to the path for imports
charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)

# Import all chart modules
from training_accuracy import TrainingAccuracyChart
from validation_accuracy import ValidationAccuracyChart
from training_loss import TrainingLossChart
from inference_time import InferenceTimeChart
from memory_usage import MemoryUsageChart
from final_accuracy import FinalAccuracyChart
from gpu_power import GpuPowerChart
from energy_per_sample import EnergyPerSampleChart
from training_time import TrainingTimeChart
from energy_efficiency import EnergyEfficiencyChart
from model_metrics import ModelMetricsChart
from loss_curves import LossCurvesChart
from temperature_monitoring import TemperatureMonitoringChart
from hardware_utilization import HardwareUtilizationChart
from bpi_analysis import BpiAnalysisChart
from tei_comparison import TeiComparisonChart
from npi_evaluation import NPIEvaluationChart
from brain_region_activation import BrainRegionActivationChart
from cognitive_process_metrics import CognitiveProcessMetricsChart
from spike_rate import SpikeRateChart
from temporal_sparsity import TemporalSparsityChart
from active_neurons import ActiveNeuronsChart
from atic_sensitivity import AticSensitivityChart
from spike_timing import SpikeTimingChart
from temporal_binding import TemporalBindingChart
from predictive_coding import PredictiveCodingChart
from neural_synchronization import NeuralSynchronizationChart
from information_integration import InformationIntegrationChart
from theoretical_framework import TheoreticalFrameworkChart
from confidence_intervals import ConfidenceIntervalsChart
from effect_sizes import EffectSizesChart
from statistical_tests import StatisticalTestsChart
from correlation_matrix import CorrelationMatrixChart
from outlier_analysis import OutlierAnalysisChart

logger = logging.getLogger(__name__)


class BenchmarkPlotter:
    """
    Benchmark plot dispatcher that coordinates chart generation.
    Imports and calls individual chart modules.
    """
    
    def __init__(self, save_dir: str, filename_suffix: str = ""):
        """
        Initialize the benchmark plotter
    
    Args:
            save_dir: Directory to save generated charts
        """
        self.save_dir = save_dir
        self.charts_dir = os.path.join(save_dir, 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        self.filename_suffix = filename_suffix or ""
        
        # Initialize all chart instances
        self.charts = {
            'training_accuracy': TrainingAccuracyChart(save_dir),
            'validation_accuracy': ValidationAccuracyChart(save_dir),
            'training_loss': TrainingLossChart(save_dir),
            'inference_time': InferenceTimeChart(save_dir),
            'memory_usage': MemoryUsageChart(save_dir),
            'final_accuracy': FinalAccuracyChart(save_dir),
            'gpu_power': GpuPowerChart(save_dir),
            'energy_per_sample': EnergyPerSampleChart(save_dir),
            'training_time': TrainingTimeChart(save_dir),
            'energy_efficiency': EnergyEfficiencyChart(save_dir),
            'model_metrics': ModelMetricsChart(save_dir),
            'loss_curves': LossCurvesChart(save_dir),
            'temperature_monitoring': TemperatureMonitoringChart(save_dir),
            'hardware_utilization': HardwareUtilizationChart(save_dir),
            'bpi_analysis': BpiAnalysisChart(save_dir),
            'tei_comparison': TeiComparisonChart(save_dir),
            'npi_evaluation': NPIEvaluationChart(save_dir),
            'brain_region_activation': BrainRegionActivationChart(save_dir),
            'cognitive_process_metrics': CognitiveProcessMetricsChart(save_dir),
            'spike_rate': SpikeRateChart(save_dir),
            'temporal_sparsity': TemporalSparsityChart(save_dir),
            'active_neurons': ActiveNeuronsChart(save_dir),
            'atic_sensitivity': AticSensitivityChart(save_dir),
            'spike_timing': SpikeTimingChart(save_dir),
            'temporal_binding': TemporalBindingChart(save_dir),
            'predictive_coding': PredictiveCodingChart(save_dir),
            'neural_synchronization': NeuralSynchronizationChart(save_dir),
            'information_integration': InformationIntegrationChart(save_dir),
            'theoretical_framework': TheoreticalFrameworkChart(save_dir),
            'confidence_intervals': ConfidenceIntervalsChart(save_dir),
            'effect_sizes': EffectSizesChart(save_dir),
            'statistical_tests': StatisticalTestsChart(save_dir),
            'correlation_matrix': CorrelationMatrixChart(save_dir),
            'outlier_analysis': OutlierAnalysisChart(save_dir),
        }
        # Apply filename suffix to all charts
        for chart in self.charts.values():
            chart.filename_suffix = self.filename_suffix

    def generate_all_charts(self, results: Dict[str, Any]) -> None:
        """
        Generate all benchmark charts by dispatching to individual modules
        
        Args:
            results: Dictionary containing SNN and ANN benchmark results
        """
        print("🚀 Starting benchmark chart generation...")
        
        try:
            # Generate charts by calling each module
            for chart_name, chart_instance in self.charts.items():
                try:
                    print(f"📊 Generating {chart_name} chart...")
                    chart_instance.plot(results)
                except Exception as e:
                    print(f"❌ Error generating {chart_name} chart: {e}")
                    # Create fallback chart
                    chart_instance.create_fallback_chart(
                        f"{chart_name.replace('_', '_')}", 
                        f"{chart_name.replace('_', ' ').title()} Comparison"
                    )
            
            print("✅ All benchmark charts generated successfully!")
            
        except Exception as e:
            print(f"❌ Error in chart generation: {e}")
            self._create_fallback_charts()

    def _create_fallback_charts(self):
        """Create fallback charts when data is unavailable"""
        fallback_charts = [
            ('01_training_accuracy', 'Training Accuracy Comparison'),
            ('02_validation_accuracy', 'Validation Accuracy Comparison'),
            ('03_training_loss', 'Training Loss Comparison'),
            ('04_inference_time', 'Inference Time Comparison'),
            ('05_memory_usage', 'Memory Usage Over Time'),
            ('06_final_accuracy', 'Final Accuracy Comparison'),
            ('07_gpu_power', 'GPU Power Consumption'),
            ('08_energy_per_sample', 'Energy per Sample Distribution'),
            ('09_training_time', 'Training Time Comparison'),
            ('10_energy_efficiency', 'Energy Efficiency Comparison'),
            ('11_model_metrics', 'Model Metrics Comparison'),
            ('12_loss_curves', 'Loss Curves Comparison'),
            ('13_temperature_monitoring', 'Temperature Monitoring'),
            ('14_hardware_utilization', 'Hardware Utilization'),
            ('15_bpi_analysis', 'BPI Analysis'),
            ('16_tei_comparison', 'TEI Comparison'),
            ('17_npi_evaluation', 'NPI Evaluation'),
            ('18_brain_region_activation', 'Brain Region Activation'),
            ('19_cognitive_process_metrics', 'Cognitive Process Metrics'),
            ('20_spike_rate', 'Spike Rate Analysis'),
            ('21_temporal_sparsity', 'Temporal Sparsity'),
            ('22_active_neurons', 'Active Neurons'),
            ('23_atic_sensitivity', 'ATIC Sensitivity'),
            ('24_spike_timing', 'Spike Timing'),
            ('25_temporal_binding', 'Temporal Binding'),
            ('26_predictive_coding', 'Predictive Coding'),
            ('27_neural_synchronization', 'Neural Synchronization'),
            ('28_information_integration', 'Information Integration'),
            ('29_theoretical_framework', 'Theoretical Framework'),
            ('30_confidence_intervals', 'Confidence Intervals'),
            ('31_effect_sizes', 'Effect Sizes'),
            ('32_statistical_tests', 'Statistical Tests'),
            ('33_correlation_matrix', 'Correlation Matrix'),
            ('34_outlier_analysis', 'Outlier Analysis'),
        ]
        
        for filename, title in fallback_charts:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'{title}\nData Not Available', 
                       ha='center', va='center', transform=ax.transAxes, 
                    fontsize=16, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                plt.savefig(os.path.join(self.charts_dir, f'{filename}.png'), 
                   dpi=300, bbox_inches='tight')
                plt.close()
                print(f"⚠️  Created fallback chart: {filename}")
            except Exception as e:
                print(f"❌ Error creating fallback chart {filename}: {e}")


def create_benchmark_plots(results_file: str, save_dir: str, results_data: Dict[str, Any] = None, filename_suffix: str = ""):
    """
    Main function to create benchmark plots
    
    Args:
        results_file: Path to results file (optional if results_data provided)
        save_dir: Directory to save generated charts
        results_data: Direct results data (optional, takes precedence over file)
    """
    print("📊 Benchmark plot generation started")
    
    # Create plotter instance
    plotter = BenchmarkPlotter(save_dir, filename_suffix=filename_suffix)
    
    # Determine data source
    if results_data is not None:
        print("🔧 DEBUG: Using direct results data")
        print(f"🔧 DEBUG: Results data type: {type(results_data)}")
        print(f"🔧 DEBUG: Results data keys: {list(results_data.keys()) if isinstance(results_data, dict) else 'Not a dict'}")
        
        # Generate charts with direct data
        plotter.generate_all_charts(results_data)
        
    elif results_file and os.path.exists(results_file):
        print(f"🔧 DEBUG: Loading results from file: {results_file}")
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            print(f"🔧 DEBUG: Loaded results type: {type(results_data)}")
            print(f"🔧 DEBUG: Loaded results keys: {list(results_data.keys()) if isinstance(results_data, dict) else 'Not a dict'}")
            
            # Generate charts with loaded data
            plotter.generate_all_charts(results_data)
            
        except Exception as e:
            print(f"❌ Error loading results from file: {e}")
            print("⚠️  Creating fallback charts")
            plotter._create_fallback_charts()
    else:
        print("⚠️  No results data provided and no valid results file found")
        print("⚠️  Creating fallback charts")
        plotter._create_fallback_charts()
    
    print("✅ Benchmark plot generation completed") 
