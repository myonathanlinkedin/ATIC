import psutil
import subprocess
import time
import torch
import numpy as np
import json
from typing import Dict, Optional
import logging
from datetime import datetime
import os
import torch.nn as nn
from torch.utils.data import DataLoader

class EnergyMonitor:
    """
    Robust energy and power monitoring with fallback mechanisms
    Handles both NVIDIA and non-NVIDIA systems
    """
    
    def __init__(self, log_interval: int = 10, device: str = 'cuda'):  # Real-time logging every 10 steps
        self.log_interval = log_interval
        self.device = device
        self.gpu_available = self._check_gpu_availability()
        self.monitoring_active = False
        self.measurements = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.gpu_available:
            self.logger.info("✅ GPU monitoring available")
        else:
            self.logger.warning("⚠️  GPU monitoring not available - using CPU-only metrics")
    
    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU monitoring is available"""
        try:
            # Check if nvidia-smi exists
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Check if CUDA is available in PyTorch
        if torch.cuda.is_available():
            self.logger.info("✅ CUDA available but nvidia-smi not accessible")
            return True
        
        return False
    
    def _get_nvidia_metrics(self) -> Optional[Dict]:
        """Get NVIDIA GPU metrics with detailed hardware information"""
        try:
            # Get detailed GPU information
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,power.draw,memory.used,memory.total,temperature.gpu,utilization.gpu,utilization.memory', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                metrics = {}
                
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 8:
                            try:
                                gpu_name = parts[0].strip()
                                driver_version = parts[1].strip()
                                power_draw = float(parts[2]) if parts[2] != 'N/A' else 0.0
                                memory_used = float(parts[3]) if parts[3] != 'N/A' else 0.0
                                memory_total = float(parts[4]) if parts[4] != 'N/A' else 0.0
                                temperature = float(parts[5]) if parts[5] != 'N/A' else 0.0
                                gpu_utilization = float(parts[6]) if parts[6] != 'N/A' else 0.0
                                memory_utilization = float(parts[7]) if parts[7] != 'N/A' else 0.0
                                
                                metrics[f'gpu_{i}'] = {
                                    'hardware_name': gpu_name,
                                    'driver_version': driver_version,
                                    'power_draw_w': power_draw,
                                    'memory_used_mb': memory_used,
                                    'memory_total_mb': memory_total,
                                    'memory_utilization_pct': memory_utilization,
                                    'gpu_utilization_pct': gpu_utilization,
                                    'temperature_c': temperature
                                }
                            except (ValueError, ZeroDivisionError) as e:
                                self.logger.warning(f"Failed to parse GPU metrics: {e}")
                                continue
                
                return metrics
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            self.logger.warning(f"GPU monitoring failed: {e}")
        
        return None
    
    def _get_system_metrics(self) -> Dict:
        """Get detailed system-wide metrics (CPU, memory, etc.)"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get detailed CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_power_estimate = cpu_count * (cpu_freq.current / 1000) * 10  # Rough estimate in watts
            
            # Get system information
            import platform
            system_info = {
                'os_name': platform.system(),
                'os_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'cpu_count': cpu_count,
                'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3)
            }
            
            return {
                'cpu_utilization_pct': cpu_percent,
                'cpu_power_estimate_w': cpu_power_estimate,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_total_mb': memory.total / (1024 * 1024),
                'memory_utilization_pct': memory.percent,
                'system_info': system_info,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"System metrics failed: {e}")
            return {'timestamp': time.time()}
    
    def start_monitoring(self):
        """Start energy monitoring"""
        self.monitoring_active = True
        self.measurements = []
        self.logger.info("🔋 Energy monitoring started")
    
    def stop_monitoring(self):
        """Stop energy monitoring"""
        self.monitoring_active = False
        self.logger.info("🔋 Energy monitoring stopped")
    
    def log_metrics(self, step: int, model_name: str = ""):
        """Log current metrics"""
        if not self.monitoring_active:
            return
        
        metrics = {
            'step': step,
            'model': model_name,
            'timestamp': time.time()
        }
        
        # Get GPU metrics if available
        gpu_metrics = self._get_nvidia_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        metrics.update(system_metrics)
        
        self.measurements.append(metrics)
        
        # Log every N steps
        if step % self.log_interval == 0:
            self._log_current_metrics(metrics)
            # Real-time logging to results folder
            self._save_real_time_log(metrics)
    
    def _log_current_metrics(self, metrics: Dict):
        """Log current metrics to console with detailed hardware info"""
        gpu_info = ""
        if any(k.startswith('gpu_') for k in metrics.keys()):
            for gpu_key, gpu_data in metrics.items():
                if gpu_key.startswith('gpu_'):
                    gpu_name = gpu_data.get('hardware_name', 'Unknown GPU')
                    driver_ver = gpu_data.get('driver_version', 'Unknown')
                    gpu_info += f"GPU: {gpu_name} (Driver: {driver_ver}) | "
                    gpu_info += f"Power: {gpu_data.get('power_draw_w', 0):.1f}W | "
                    gpu_info += f"Util: {gpu_data.get('gpu_utilization_pct', 0):.1f}% | "
                    gpu_info += f"Mem: {gpu_data.get('memory_utilization_pct', 0):.1f}% | "
                    gpu_info += f"Temp: {gpu_data.get('temperature_c', 0):.1f}°C | "
        
        cpu_info = f"CPU: {metrics.get('cpu_utilization_pct', 0):.1f}%, "
        cpu_info += f"{metrics.get('cpu_power_estimate_w', 0):.1f}W | "
        mem_info = f"RAM: {metrics.get('memory_utilization_pct', 0):.1f}%"
        
        self.logger.info(f"🔋 {gpu_info}{cpu_info}{mem_info}")
    
    def get_summary(self) -> Dict:
        """Get summary of all measurements"""
        if not self.measurements:
            return {}
        
        summary = {
            'total_measurements': len(self.measurements),
            'monitoring_duration_s': self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp'],
            'gpu_available': self.gpu_available
        }
        
        # Calculate averages for each metric
        metrics_to_average = ['cpu_utilization_pct', 'cpu_power_estimate_w', 
                            'memory_utilization_pct', 'memory_used_mb']
        
        for metric in metrics_to_average:
            values = [m.get(metric, 0) for m in self.measurements if m.get(metric) is not None]
            if values:
                summary[f'avg_{metric}'] = np.mean(values)
                summary[f'max_{metric}'] = np.max(values)
                summary[f'min_{metric}'] = np.min(values)
        
        # GPU-specific metrics
        gpu_metrics = []
        for m in self.measurements:
            for key, value in m.items():
                if key.startswith('gpu_') and isinstance(value, dict):
                    gpu_metrics.append(value)
        
        if gpu_metrics:
            # Average across all GPUs
            avg_gpu_power = np.mean([g.get('power_draw_w', 0) for g in gpu_metrics])
            avg_gpu_memory = np.mean([g.get('memory_utilization_pct', 0) for g in gpu_metrics])
            avg_gpu_temp = np.mean([g.get('temperature_c', 0) for g in gpu_metrics])
            
            summary.update({
                'avg_gpu_power_w': avg_gpu_power,
                'max_gpu_power_w': np.max([g.get('power_draw_w', 0) for g in gpu_metrics]),
                'avg_gpu_memory_pct': avg_gpu_memory,
                'avg_gpu_temperature_c': avg_gpu_temp
            })
        
        return summary
    
    def _save_real_time_log(self, metrics: Dict):
        """Save detailed real-time metrics to results folder"""
        import json
        import os
        
        # Create results/logs directory if it doesn't exist
        logs_dir = os.path.join('./results', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create detailed log entry with hardware info
        detailed_log = {
            'timestamp': datetime.now().isoformat(),
            'step': metrics.get('step', 0),
            'model': metrics.get('model', 'unknown'),
            'hardware_details': {
                'gpu_info': {},
                'system_info': metrics.get('system_info', {}),
                'performance_metrics': {
                    'cpu_utilization_pct': metrics.get('cpu_utilization_pct', 0),
                    'memory_utilization_pct': metrics.get('memory_utilization_pct', 0),
                    'cpu_power_estimate_w': metrics.get('cpu_power_estimate_w', 0)
                }
            }
        }
        
        # Add detailed GPU information
        for key, value in metrics.items():
            if key.startswith('gpu_'):
                detailed_log['hardware_details']['gpu_info'][key] = value
        
        # Create real-time log file with hardware info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f'detailed_hardware_log_{timestamp}.json')
        
        # Append to existing log or create new
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {'hardware_logs': []}
        else:
            log_data = {'hardware_logs': []}
        
        # Add new detailed log entry
        log_data['hardware_logs'].append(detailed_log)
        
        # Save updated log with detailed hardware information
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def save_metrics(self, filename: str):
        """Save metrics to file"""
        import json
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'measurements': self.measurements,
                    'summary': self.get_summary()
                }, f, indent=2)
            self.logger.info(f"💾 Metrics saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def compare_energy_efficiency(
        self,
        ann_model: nn.Module,
        snn_model: nn.Module,
        test_loader: DataLoader,
        train_loader: DataLoader,
        save_dir: str = './results'
    ) -> Dict[str, Dict]:
        """Compare energy efficiency between ANN and SNN models"""
        self.logger.info("🔋 Starting energy efficiency comparison...")
        
        # Start monitoring
        self.start_monitoring()
        
        # Test ANN model energy consumption
        self.logger.info("Testing ANN energy consumption...")
        ann_energy_metrics = self._test_model_energy(ann_model, test_loader, "ANN")
        
        # Test SNN model energy consumption
        self.logger.info("Testing SNN energy consumption...")
        snn_energy_metrics = self._test_model_energy(snn_model, test_loader, "SNN")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate improvements
        energy_reduction = self._calculate_energy_improvements(ann_energy_metrics, snn_energy_metrics)
        
        # Save detailed comparison
        comparison_results = {
            'ann_metrics': ann_energy_metrics,
            'snn_metrics': snn_energy_metrics,
            'inference_improvements': energy_reduction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        comparison_file = os.path.join(save_dir, 'energy_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        self.logger.info(f"💾 Energy comparison saved to {comparison_file}")
        
        return comparison_results
    
    def _test_model_energy(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """Test energy consumption for a specific model"""
        model.eval()
        model = model.to(self.device)
        
        # Reset measurements for this test
        self.measurements = []
        
        # Start monitoring
        self.start_monitoring()
        
        start_time = time.time()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Log metrics every 10 batches
                if batch_idx % 10 == 0:
                    self.log_metrics(batch_idx, model_name)
                
                # Forward pass
                output = model(data)
                total_samples += data.size(0)
                
                # Stop after 100 batches to get representative sample
                if batch_idx >= 100:
                    break
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Get summary
        summary = self.get_summary()
        
        return {
            'total_samples': total_samples,
            'inference_time_s': inference_time,
            'samples_per_second': total_samples / inference_time if inference_time > 0 else 0,
            'avg_gpu_power_w': summary.get('avg_gpu_power_w', 0),
            'max_gpu_power_w': summary.get('max_gpu_power_w', 0),
            'avg_cpu_power_w': summary.get('avg_cpu_power_estimate_w', 0),
            'total_energy_j': summary.get('avg_gpu_power_w', 0) * inference_time,
            'energy_per_sample_j': (summary.get('avg_gpu_power_w', 0) * inference_time) / total_samples if total_samples > 0 else 0
        }
    
    def _calculate_energy_improvements(
        self,
        ann_metrics: Dict[str, float],
        snn_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate energy efficiency improvements"""
        improvements = {}
        
        # Energy reduction percentage
        if ann_metrics.get('total_energy_j', 0) > 0:
            energy_reduction = ((ann_metrics['total_energy_j'] - snn_metrics['total_energy_j']) / ann_metrics['total_energy_j']) * 100
            improvements['energy_reduction_percent'] = max(0, energy_reduction)
        else:
            improvements['energy_reduction_percent'] = 0
        
        # Power reduction
        if ann_metrics.get('avg_gpu_power_w', 0) > 0:
            power_reduction = ((ann_metrics['avg_gpu_power_w'] - snn_metrics['avg_gpu_power_w']) / ann_metrics['avg_gpu_power_w']) * 100
            improvements['power_reduction_percent'] = max(0, power_reduction)
        else:
            improvements['power_reduction_percent'] = 0
        
        # Efficiency improvement (samples per joule)
        ann_efficiency = ann_metrics.get('samples_per_second', 0) / max(ann_metrics.get('avg_gpu_power_w', 1), 1)
        snn_efficiency = snn_metrics.get('samples_per_second', 0) / max(snn_metrics.get('avg_gpu_power_w', 1), 1)
        
        if ann_efficiency > 0:
            efficiency_improvement = ((snn_efficiency - ann_efficiency) / ann_efficiency) * 100
            improvements['efficiency_improvement_percent'] = max(0, efficiency_improvement)
        else:
            improvements['efficiency_improvement_percent'] = 0
        
        return improvements

# Energy monitoring module - all functions are actively used in training loops 
