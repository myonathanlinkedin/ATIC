"""
Comprehensive error handling system for SNN vs ANN benchmark
Provides detailed error messages and recovery mechanisms
"""

import sys
import traceback
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import torch
import os


class BenchmarkError(Exception):
    """Base exception class for benchmark errors"""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DatasetError(BenchmarkError):
    """Exception for dataset-related errors"""
    pass


class ModelError(BenchmarkError):
    """Exception for model-related errors"""
    pass


class HardwareError(BenchmarkError):
    """Exception for hardware-related errors"""
    pass


class TrainingError(BenchmarkError):
    """Exception for training-related errors"""
    pass


class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file) if self.log_file 
                else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_dataset_error(self, error: Exception, dataset_name: str) -> str:
        """Handle dataset loading errors with detailed guidance"""
        if isinstance(error, FileNotFoundError):
            message = f"❌ Dataset '{dataset_name}' not found!\n"
            message += "📥 Please download manually from:\n"
            if dataset_name.lower() == 'nmnist':
                message += "   https://www.garrickorchard.com/datasets/n-mnist\n"
            elif dataset_name.lower() == 'shd':
                message += "   https://zenkelab.org/resources/\n"
            message += f"📁 Extract to: ./data/{dataset_name}/\n"
            message += "🔄 Then run the benchmark again."
            
            self.logger.error(f"Dataset error: {error}")
            return message
        else:
            return f"❌ Dataset error: {str(error)}"
    
    def handle_model_error(self, error: Exception, model_type: str) -> str:
        """Handle model creation/loading errors"""
        if isinstance(error, RuntimeError) and "CUDA" in str(error):
            message = f"❌ CUDA error with {model_type} model!\n"
            message += "💡 Try using CPU: --device cpu\n"
            message += "💡 Or reduce batch size: --batch-size 32\n"
            message += "💡 Or check GPU memory: nvidia-smi"
            
            self.logger.error(f"Model error: {error}")
            return message
        else:
            return f"❌ Model error: {str(error)}"
    
    def handle_memory_error(self, error: Exception) -> str:
        """Handle GPU memory errors with solutions"""
        message = "❌ GPU memory insufficient!\n"
        message += "💡 Solutions:\n"
        message += "   1. Reduce batch size: --batch-size 32\n"
        message += "   2. Use CPU: --device cpu\n"
        message += "   3. Clear GPU memory: nvidia-smi --gpu-reset\n"
        message += "   4. Restart your computer\n"
        
        self.logger.error(f"Memory error: {error}")
        return message
    
    def handle_training_error(self, error: Exception, epoch: int = None) -> str:
        """Handle training errors with recovery suggestions"""
        message = "❌ Training error"
        if epoch:
            message += f" at epoch {epoch}"
        message += "!\n"
        
        if "loss" in str(error).lower():
            message += "💡 Try:\n"
            message += "   1. Reduce learning rate: --learning-rate 1e-4\n"
            message += "   2. Increase epochs: --epochs 30\n"
            message += "   3. Check data quality\n"
        elif "gradient" in str(error).lower():
            message += "💡 Try:\n"
            message += "   1. Reduce learning rate: --learning-rate 1e-4\n"
            message += "   2. Use gradient clipping\n"
            message += "   3. Check for NaN values in data\n"
        else:
            message += "💡 Try:\n"
            message += "   1. Restart training: --force-retrain\n"
            message += "   2. Use CPU: --device cpu\n"
            message += "   3. Check system resources\n"
        
        self.logger.error(f"Training error: {error}")
        return message
    
    def handle_hardware_error(self, error: Exception) -> str:
        """Handle hardware-related errors"""
        if "temperature" in str(error).lower():
            message = "❌ GPU temperature too high!\n"
            message += "💡 Solutions:\n"
            message += "   1. Improve cooling/ventilation\n"
            message += "   2. Reduce GPU load: --batch-size 16\n"
            message += "   3. Use CPU: --device cpu\n"
            message += "   4. Wait for GPU to cool down\n"
        elif "power" in str(error).lower():
            message = "❌ GPU power limit exceeded!\n"
            message += "💡 Solutions:\n"
            message += "   1. Reduce batch size: --batch-size 16\n"
            message += "   2. Use CPU: --device cpu\n"
            message += "   3. Check power supply\n"
        else:
            message = f"❌ Hardware error: {str(error)}\n"
            message += "💡 Try using CPU: --device cpu"
        
        self.logger.error(f"Hardware error: {error}")
        return message
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check if system meets requirements"""
        checks = {
            'cuda_available': torch.cuda.is_available(),
            'sufficient_memory': self._check_memory(),
            'disk_space': self._check_disk_space(),
            'python_version': self._check_python_version()
        }
        
        return checks
    
    def _check_memory(self) -> bool:
        """Check if system has sufficient memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.available > 4 * 1024 * 1024 * 1024  # 4GB
        except:
            return True  # Assume OK if can't check
    
    def _check_disk_space(self) -> bool:
        """Check if there's sufficient disk space"""
        try:
            stat = os.statvfs('.')
            free_space = stat.f_frsize * stat.f_bavail
            return free_space > 2 * 1024 * 1024 * 1024  # 2GB
        except:
            return True  # Assume OK if can't check
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        return sys.version_info >= (3, 8)
    
    def print_system_status(self):
        """Print comprehensive system status"""
        checks = self.check_system_requirements()
        
        print("🔍 System Requirements Check:")
        print("=" * 40)
        
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check.replace('_', ' ').title()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA GPU: Not available")
        
        print("=" * 40)


def error_handler(func: Callable) -> Callable:
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = ErrorHandler()
        
        try:
            return func(*args, **kwargs)
        except DatasetError as e:
            print(handler.handle_dataset_error(e, 
                  e.details.get('dataset_name', 'Unknown')))
            sys.exit(1)
        except ModelError as e:
            print(handler.handle_model_error(e, 
                  e.details.get('model_type', 'Unknown')))
            sys.exit(1)
        except TrainingError as e:
            print(handler.handle_training_error(e, e.details.get('epoch')))
            sys.exit(1)
        except HardwareError as e:
            print(handler.handle_hardware_error(e))
            sys.exit(1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(handler.handle_memory_error(e))
            elif "cuda" in str(e).lower():
                print(handler.handle_hardware_error(e))
            else:
                print(f"❌ Runtime error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            print("📋 Full traceback:")
            traceback.print_exc()
            sys.exit(1)
    
    return wrapper


def safe_execute(func: Callable, error_message: str = None) -> Optional[Any]:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        handler = ErrorHandler()
        handler.logger.error(f"Safe execute error: {e}")
        
        if error_message:
            print(f"❌ {error_message}")
        else:
            print(f"❌ Error: {str(e)}")
        
        return None


def validate_inputs(**kwargs) -> bool:
    """Validate input parameters"""
    handler = ErrorHandler()
    
    # Check batch size
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
        if batch_size <= 0:
            print("❌ Batch size must be positive!")
            return False
        if batch_size > 256:
            print("⚠️  Large batch size may cause memory issues!")
    
    # Check learning rate
    if 'learning_rate' in kwargs:
        lr = kwargs['learning_rate']
        if lr <= 0 or lr > 1:
            print("❌ Learning rate must be between 0 and 1!")
            return False
    
    # Check epochs
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
        if epochs <= 0:
            print("❌ Number of epochs must be positive!")
            return False
        if epochs > 1000:
            print("⚠️  Large number of epochs may take very long!")
    
    # Check device
    if 'device' in kwargs:
        device = kwargs['device']
        if device == 'cuda' and not torch.cuda.is_available():
            print("❌ CUDA requested but not available!")
            return False
    
    return True


def print_success_message(message: str, details: Dict = None):
    """Print standardized success messages"""
    print(f"✅ {message}")
    if details:
        for key, value in details.items():
            print(f"   {key}: {value}")


def print_warning_message(message: str, suggestion: str = None):
    """Print standardized warning messages"""
    print(f"⚠️  {message}")
    if suggestion:
        print(f"💡 Suggestion: {suggestion}")


def print_error_message(message: str, solution: str = None):
    """Print standardized error messages"""
    print(f"❌ {message}")
    if solution:
        print(f"💡 Solution: {solution}") 
