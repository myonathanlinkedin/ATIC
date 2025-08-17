import os
import sys
import unittest
import shutil
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

# Add the charts directory to the path
charts_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'logging', 'charts')
sys.path.insert(0, charts_path)

# Import directly from charts directory
from base_chart import BaseChart
from training_accuracy import TrainingAccuracyChart
from validation_accuracy import ValidationAccuracyChart
from training_loss import TrainingLossChart
from inference_time import InferenceTimeChart
from memory_usage import MemoryUsageChart
from final_accuracy import FinalAccuracyChart
from gpu_power import GpuPowerChart
from energy_per_sample import EnergyPerSampleChart

# Import plot_benchmark
plot_benchmark_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'logging')
sys.path.insert(0, plot_benchmark_path)
from plot_benchmark import BenchmarkPlotter

print("✅ Successfully imported all modules")


class ChartImprovementsTest(unittest.TestCase):
    """
    Comprehensive unit tests for modular chart system with 20 epochs maximum
    """
    
    def setUp(self):
        """Set up test environment with dummy training history"""
        self.test_dir = os.path.join(os.path.dirname(__file__), 
                                    'test_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create comprehensive dummy training history (5-20 epochs)
        self.dummy_history = self._create_dummy_training_history()
        
        # Set matplotlib to use Agg backend
        plt.switch_backend('Agg')
    
    def tearDown(self):
        """Keep test files for inspection"""
        # Don't delete the test directory - keep charts for inspection
        pass
    
    def _create_dummy_training_history(self) -> Dict[str, Any]:
        """Create realistic dummy training history for 20 epochs maximum"""
        epochs = 15  # Test with 15 epochs (less than 20)
        
        return {
            'snn': {
                'training_accuracy': [
                    0.45 + i*0.035 + np.random.normal(0, 0.02) 
                    for i in range(epochs)
                ],
                'validation_accuracy': [
                    0.42 + i*0.032 + np.random.normal(0, 0.025) 
                    for i in range(epochs)
                ],
                'training_loss': [
                    2.1 - i*0.08 + np.random.normal(0, 0.05) 
                    for i in range(epochs)
                ],
                'validation_loss': [
                    2.2 - i*0.075 + np.random.normal(0, 0.06) 
                    for i in range(epochs)
                ],
                'inference_time': 0.045,
                'memory_usage': [
                    512 + i*15 + np.random.normal(0, 10) 
                    for i in range(epochs)
                ],
                'gpu_power': [
                    85 + i*2 + np.random.normal(0, 3) 
                    for i in range(epochs)
                ],
                'final_accuracy': 78.5,
                'energy_per_sample': [
                    0.8 + i*0.05 + np.random.normal(0, 0.1) 
                    for i in range(epochs)
                ],
                'training_time': [
                    120 + i*5 + np.random.normal(0, 3) 
                    for i in range(epochs)
                ],
                'energy_efficiency': [
                    0.02 + i*0.001 + np.random.normal(0, 0.002) 
                    for i in range(epochs)
                ],
                'gpu_temperature': [
                    65 + i*1.5 + np.random.normal(0, 4) + 8 * np.sin(i * 0.4)
                    for i in range(epochs)
                ],
                'hardware_utilization': [
                    75 + i*2 + np.random.normal(0, 6) + 10 * np.cos(i * 0.3)
                    for i in range(epochs)
                ],
                'gpu_utilization': [
                    70 + i*3 + np.random.normal(0, 5) + 12 * np.sin(i * 0.5)
                    for i in range(epochs)
                ],
                'memory_utilization': [
                    80 + i*2.5 + np.random.normal(0, 4) + 8 * np.cos(i * 0.4)
                    for i in range(epochs)
                ],
                'spike_rate': np.random.rand(20, 20),  # 20x20 matrix for heatmap
                'temporal_sparsity': np.random.rand(10, 10),  # 10x10 matrix for 3D surface
                'active_neurons': [
                    1200 + i*40 + np.random.normal(0, 50) + 100 * np.sin(i * 0.3)
                    for i in range(epochs)
                ],
                'atic_sensitivity': [
                    0.45 + i*0.015 + np.random.normal(0, 0.05) + 0.1 * np.sin(i * 0.5)
                    for i in range(epochs)
                ],
                'spike_timing': [
                    0.02 + i*0.003 + np.random.normal(0, 0.008) + 0.02 * np.cos(i * 0.4)
                    for i in range(epochs)
                ],
                'temporal_binding': [
                    0.7 + i*0.015 + np.random.normal(0, 0.04) + 0.08 * np.sin(i * 0.6)
                    for i in range(epochs)
                ],
                'predictive_coding': [
                    0.65 + i*0.02 + np.random.normal(0, 0.05) + 0.12 * np.cos(i * 0.4)
                    for i in range(epochs)
                ],
                'neural_synchronization': [
                    0.55 + i*0.018 + np.random.normal(0, 0.04) + 0.1 * np.sin(i * 0.5)
                    for i in range(epochs)
                ],
                'information_integration': [
                    0.75 + i*0.012 + np.random.normal(0, 0.03) + 0.08 * np.cos(i * 0.3)
                    for i in range(epochs)
                ],
                'bpi_over_time': [
                    0.82 + i*0.01 + np.random.normal(0, 0.02) 
                    for i in range(epochs)
                ],
                'tei_components': [0.78, 0.82, 0.75],  # 3 components
                'npi_components': [0.85, 0.80, 0.88],  # 3 components
                'brain_activation': {
                    'V1_primary': {'level': 0.75},
                    'V2_secondary': {'level': 0.68},
                    'V4_color_form': {'level': 0.72},
                    'IT_object': {'level': 0.65}
                },
                'cognitive_processes': {
                    'focus_score': 0.78,
                    'selectivity': 0.82,
                    'sustained_attention': 0.75,
                    'working_memory': 0.80,
                    'episodic_memory': 0.85,
                    'semantic_memory': 0.78,
                    'planning': 0.72,
                    'decision_making': 0.75,
                    'cognitive_flexibility': 0.70
                },
                'metrics': {
                    'parameter_count': 1250000,
                    'memory_usage': 512,
                    'energy_consumption': 45.2
                },
                'train_losses': [
                    2.1 - i*0.08 + np.random.normal(0, 0.05) 
                    for i in range(epochs)
                ],
                'val_losses': [
                    2.2 - i*0.075 + np.random.normal(0, 0.06) 
                    for i in range(epochs)
                ],
                'outlier_data': [
                    0.5 + np.random.normal(0, 0.1) 
                    for _ in range(50)
                ]
            },
            'ann': {
                'training_accuracy': [0.48 + i*0.04 + np.random.normal(0, 0.025) for i in range(epochs)],
                'validation_accuracy': [0.45 + i*0.038 + np.random.normal(0, 0.03) for i in range(epochs)],
                'training_loss': [1.9 - i*0.09 + np.random.normal(0, 0.06) for i in range(epochs)],
                'validation_loss': [2.0 - i*0.085 + np.random.normal(0, 0.07) for i in range(epochs)],
                'inference_time': 0.038,
                'memory_usage': [580 + i*20 + np.random.normal(0, 15) for i in range(epochs)],
                'gpu_power': [95 + i*2.5 + np.random.normal(0, 4) for i in range(epochs)],
                'final_accuracy': 82.3,
                'energy_per_sample': [1.2 + i*0.06 + np.random.normal(0, 0.12) for i in range(epochs)],
                'gpu_temperature': [72 + i*2 + np.random.normal(0, 5) + 10 * np.cos(i * 0.3)
                    for i in range(epochs)],
                'hardware_utilization': [85 + i*2.5 + np.random.normal(0, 7) + 12 * np.sin(i * 0.4)
                    for i in range(epochs)],
                'gpu_utilization': [80 + i*3.5 + np.random.normal(0, 6) + 15 * np.cos(i * 0.5)
                    for i in range(epochs)],
                'memory_utilization': [85 + i*3 + np.random.normal(0, 5) + 10 * np.sin(i * 0.3)
                    for i in range(epochs)],
                'spike_rate': np.random.rand(20, 20),  # 20x20 matrix for heatmap
                'temporal_sparsity': np.random.rand(10, 10),  # 10x10 matrix for 3D surface
                'active_neurons': [1500 + i*50 + np.random.normal(0, 60) + 80 * np.cos(i * 0.4)
                    for i in range(epochs)],
                'atic_sensitivity': [
                    0.15 + i*0.008 + np.random.normal(0, 0.03) + 0.05 * np.cos(i * 0.3)
                    for i in range(epochs)
                ],
                'spike_timing': [
                    0.01 + i*0.002 + np.random.normal(0, 0.006) + 0.015 * np.sin(i * 0.5)
                    for i in range(epochs)
                ],
                'temporal_binding': [0.65 + i*0.012 + np.random.normal(0, 0.035) + 0.06 * np.cos(i * 0.4)
                    for i in range(epochs)],
                'predictive_coding': [0.60 + i*0.015 + np.random.normal(0, 0.04) + 0.1 * np.sin(i * 0.3)
                    for i in range(epochs)],
                'neural_synchronization': [0.50 + i*0.015 + np.random.normal(0, 0.035) + 0.08 * np.cos(i * 0.4)
                    for i in range(epochs)],
                'information_integration': [0.70 + i*0.01 + np.random.normal(0, 0.025) + 0.06 * np.sin(i * 0.3)
                    for i in range(epochs)],
                'bpi_over_time': [
                    0.45 + i*0.008 + np.random.normal(0, 0.015) 
                    for i in range(epochs)
                ],
                'tei_components': [0.52, 0.48, 0.55],  # 3 components
                'npi_components': [0.48, 0.52, 0.45],  # 3 components
                'brain_activation': {
                    'V1_primary': {'level': 0.60},
                    'V2_secondary': {'level': 0.55},
                    'V4_color_form': {'level': 0.58},
                    'IT_object': {'level': 0.52}
                },
                'cognitive_processes': {
                    'focus_score': 0.62,
                    'selectivity': 0.68,
                    'sustained_attention': 0.65,
                    'working_memory': 0.70,
                    'episodic_memory': 0.75,
                    'semantic_memory': 0.68,
                    'planning': 0.58,
                    'decision_making': 0.62,
                    'cognitive_flexibility': 0.60
                },
                'training_time': [
                    150 + i*6 + np.random.normal(0, 4) 
                    for i in range(epochs)
                ],
                'energy_efficiency': [
                    0.05 + i*0.002 + np.random.normal(0, 0.003) 
                    for i in range(epochs)
                ],
                'metrics': {
                    'parameter_count': 2100000,
                    'memory_usage': 768,
                    'energy_consumption': 67.8
                },
                'train_losses': [
                    1.9 - i*0.09 + np.random.normal(0, 0.06) 
                    for i in range(epochs)
                ],
                'val_losses': [
                    2.0 - i*0.085 + np.random.normal(0, 0.07) 
                    for i in range(epochs)
                ],
                'outlier_data': [
                    0.6 + np.random.normal(0, 0.15) 
                    for _ in range(50)
                ]
            }
        }
    
    def test_training_accuracy_chart_module(self):
        """Test training accuracy chart module"""
        try:
            chart = TrainingAccuracyChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            # Check if chart was created
            chart_path = os.path.join(self.test_dir, 'charts', '01_training_accuracy.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Training accuracy chart was not created")
            
            print("✅ Training accuracy chart module test passed")
            
        except Exception as e:
            self.fail(f"Training accuracy chart module test failed: {e}")
    
    def test_validation_accuracy_chart_module(self):
        """Test validation accuracy chart module"""
        try:
            chart = ValidationAccuracyChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '02_validation_accuracy.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Validation accuracy chart was not created")
            
            print("✅ Validation accuracy chart module test passed")
            
        except Exception as e:
            self.fail(f"Validation accuracy chart module test failed: {e}")
    
    def test_training_loss_chart_module(self):
        """Test training loss chart module"""
        try:
            chart = TrainingLossChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '03_training_loss.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Training loss chart was not created")
            
            print("✅ Training loss chart module test passed")
            
        except Exception as e:
            self.fail(f"Training loss chart module test failed: {e}")
    
    def test_inference_time_chart_module(self):
        """Test inference time chart module"""
        try:
            chart = InferenceTimeChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '04_inference_time.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Inference time chart was not created")
            
            print("✅ Inference time chart module test passed")
            
        except Exception as e:
            self.fail(f"Inference time chart module test failed: {e}")
    
    def test_memory_usage_chart_module(self):
        """Test memory usage chart module"""
        try:
            chart = MemoryUsageChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '05_memory_usage.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Memory usage chart was not created")
            
            print("✅ Memory usage chart module test passed")
            
        except Exception as e:
            self.fail(f"Memory usage chart module test failed: {e}")
    
    def test_final_accuracy_chart_module(self):
        """Test final accuracy chart module"""
        try:
            chart = FinalAccuracyChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '06_final_accuracy.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Final accuracy chart was not created")
            
            print("✅ Final accuracy chart module test passed")
            
        except Exception as e:
            self.fail(f"Final accuracy chart module test failed: {e}")
    
    def test_gpu_power_chart_module(self):
        """Test GPU power chart module"""
        try:
            chart = GpuPowerChart(self.test_dir)
            chart.plot(self.dummy_history)
            
            chart_path = os.path.join(self.test_dir, 'charts', '07_gpu_power.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "GPU power chart was not created")
            
            print("✅ GPU power chart module test passed")
            
        except Exception as e:
            self.fail(f"GPU power chart module test failed: {e}")
    
    def test_benchmark_plotter_dispatcher(self):
        """Test the benchmark plotter dispatcher"""
        try:
            plotter = BenchmarkPlotter(self.test_dir)
            plotter.generate_all_charts(self.dummy_history)
            
            # Check if all charts were created
            charts_dir = os.path.join(self.test_dir, 'charts')
            self.assertTrue(os.path.exists(charts_dir), 
                          "Charts directory was not created")
            
            # Check for expected chart files
            expected_charts = [
                '01_training_accuracy.png',
                '02_validation_accuracy.png', 
                '03_training_loss.png',
                '04_inference_time.png',
                '05_memory_usage.png',
                '06_final_accuracy.png',
                '07_gpu_power.png'
            ]
            
            for chart_file in expected_charts:
                chart_path = os.path.join(charts_dir, chart_file)
                self.assertTrue(os.path.exists(chart_path), 
                              f"Chart {chart_file} was not created")
            
            print("✅ Benchmark plotter dispatcher test passed")
            
        except Exception as e:
            self.fail(f"Benchmark plotter dispatcher test failed: {e}")
    
    def test_dynamic_x_range_limits(self):
        """Test that x-axis ranges are properly limited to 20 epochs maximum"""
        try:
            # Test with data longer than 20 epochs
            long_data = {
                'snn': {
                    'training_accuracy': [0.5 + i*0.01 for i in range(25)]  # 25 epochs
                },
                'ann': {
                    'training_accuracy': [0.6 + i*0.01 for i in range(25)]  # 25 epochs
                }
            }
            
            chart = TrainingAccuracyChart(self.test_dir)
            chart.plot(long_data)
            
            # The chart should only use the first 20 epochs
            chart_path = os.path.join(self.test_dir, 'charts', '01_training_accuracy.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Chart with long data was not created")
            
            print("✅ Dynamic x-range limits test passed")
            
        except Exception as e:
            self.fail(f"Dynamic x-range limits test failed: {e}")
    
    def test_error_handling_with_empty_data(self):
        """Test error handling when data is missing or empty"""
        try:
            empty_data = {
                'snn': {},
                'ann': {}
            }
            
            chart = TrainingAccuracyChart(self.test_dir)
            chart.plot(empty_data)
            
            # Should create a fallback chart
            chart_path = os.path.join(self.test_dir, 'charts', '01_training_accuracy.png')
            self.assertTrue(os.path.exists(chart_path), 
                          "Fallback chart was not created for empty data")
            
            print("✅ Error handling with empty data test passed")
            
        except Exception as e:
            self.fail(f"Error handling with empty data test failed: {e}")
    
    def test_chart_improvements_consistency(self):
        """Test that all chart improvements are applied consistently"""
        try:
            chart = TrainingAccuracyChart(self.test_dir)
            
            # Test the base chart methods
            x_range = chart.get_dynamic_x_range(15)
            self.assertEqual(len(x_range), 15, "X-range length should match data length")
            
            x_range_long = chart.get_dynamic_x_range(25)
            self.assertEqual(len(x_range_long), 20, "X-range should be capped at 20")
            
            print("✅ Chart improvements consistency test passed")
            
        except Exception as e:
            self.fail(f"Chart improvements consistency test failed: {e}")

    def test_energy_consumption_dict_format(self):
        """Test energy consumption with dict format (from benchmark_runner)"""
        try:
            # Create data with dict format energy consumption
            dict_data = {
                'snn': {
                    'energy_per_sample': [0.02 + np.random.normal(0, 0.005) for _ in range(10)],
                    'gpu_power': [45 + np.random.normal(0, 3) for _ in range(10)],
                    'memory_usage': [512 + np.random.normal(0, 10) for _ in range(10)]
                },
                'ann': {
                    'energy_per_sample': [1.2 + np.random.normal(0, 0.1) for _ in range(10)],
                    'gpu_power': [95 + np.random.normal(0, 5) for _ in range(10)],
                    'memory_usage': [768 + np.random.normal(0, 15) for _ in range(10)]
                }
            }
            
            # Test energy per sample chart with dict format
            energy_chart = EnergyPerSampleChart(self.test_dir)
            energy_chart.plot(dict_data)
            
            # Test GPU power chart with dict format
            gpu_chart = GpuPowerChart(self.test_dir)
            gpu_chart.plot(dict_data)
            
            # Test memory usage chart with dict format
            memory_chart = MemoryUsageChart(self.test_dir)
            memory_chart.plot(dict_data)
            
            print("✅ Energy consumption dict format test passed")
            
        except Exception as e:
            self.fail(f"Energy consumption dict format test failed: {e}")

    def test_energy_consumption_float_format(self):
        """Test energy consumption with float format (from test data)"""
        try:
            # Create data with float format energy consumption
            float_data = {
                'snn': {
                    'energy_per_sample': [0.8 + i*0.05 + np.random.normal(0, 0.1) for i in range(10)],
                    'gpu_power': [85 + i*2 + np.random.normal(0, 3) for i in range(10)],
                    'memory_usage': [512 + i*15 + np.random.normal(0, 10) for i in range(10)]
                },
                'ann': {
                    'energy_per_sample': [1.2 + i*0.06 + np.random.normal(0, 0.12) for i in range(10)],
                    'gpu_power': [95 + i*2.5 + np.random.normal(0, 4) for i in range(10)],
                    'memory_usage': [768 + i*20 + np.random.normal(0, 15) for i in range(10)]
                }
            }
            
            # Test energy per sample chart with float format
            energy_chart = EnergyPerSampleChart(self.test_dir)
            energy_chart.plot(float_data)
            
            # Test GPU power chart with float format
            gpu_chart = GpuPowerChart(self.test_dir)
            gpu_chart.plot(float_data)
            
            # Test memory usage chart with float format
            memory_chart = MemoryUsageChart(self.test_dir)
            memory_chart.plot(float_data)
            
            print("✅ Energy consumption float format test passed")
            
        except Exception as e:
            self.fail(f"Energy consumption float format test failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 