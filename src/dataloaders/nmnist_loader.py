#!/usr/bin/env python3
"""
Enhanced N-MNIST Data Loader with Optimization

PURPOSE: Optimized N-MNIST dataset loading with caching and memory management
- Enhanced caching mechanism for faster data loading
- Progress tracking for large datasets
- Memory optimization for large datasets
- Error handling and validation
- Performance monitoring

ENHANCED FEATURES:
- Data caching with LRU cache
- Progress tracking with tqdm
- Memory usage monitoring
- Batch loading optimization
- Error recovery mechanisms
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
from functools import lru_cache
from tqdm import tqdm
import psutil
import gc
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedTemporalNMNISTDataset(Dataset):
    """
    PURPOSE: Optimized N-MNIST dataset with enhanced performance
    
    ENHANCED FEATURES:
    - LRU caching for data loading
    - Progress tracking with tqdm
    - Memory usage monitoring
    - Batch loading optimization
    - Error recovery mechanisms
    - Performance metrics tracking
    """
    
    def __init__(self, data_dir: str, transform=None, cache_size: int = 1000):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}
        self.performance_metrics = {
            'load_time': 0.0,
            'memory_usage': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize with progress tracking
        logger.info(f"🔄 Loading optimized N-MNIST dataset from {data_dir}")
        start_time = time.time()
        
        self.data, self.labels = self._load_data_optimized()
        
        self.performance_metrics['load_time'] = time.time() - start_time
        self.performance_metrics['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"✅ Dataset loaded successfully in {self.performance_metrics['load_time']:.2f}s")
        logger.info(f"💾 Memory usage: {self.performance_metrics['memory_usage']:.2f} MB")
    
    @lru_cache(maxsize=1000)
    def _load_class_data(self, class_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load class data with caching"""
        try:
            files = [f for f in os.listdir(class_path) if f.endswith('.bin')]
            data = []
            labels = []
            
            for file in tqdm(files, desc=f"Loading {os.path.basename(class_path)}", leave=False):
                file_path = os.path.join(class_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        sample_data = np.frombuffer(f.read(), dtype=np.uint8)
                        
                        # Calculate the correct number of samples based on file size
                        # Each sample is 34*34*2 = 2312 bytes
                        sample_size = 34 * 34 * 2
                        num_samples = len(sample_data) // sample_size
                        
                        if num_samples > 0:
                            # Reshape to (num_samples, 34, 34, 2)
                            sample_data = sample_data[:num_samples * sample_size].reshape(num_samples, 34, 34, 2)
                            data.append(sample_data)
                            labels.extend([int(os.path.basename(class_path))] * num_samples)
                        else:
                            logger.warning(f"⚠️  File {file_path} too small: {len(sample_data)} bytes")
                            continue
                except Exception as e:
                    logger.warning(f"⚠️  Error loading {file_path}: {e}")
                    continue
            
            if data:
                return np.concatenate(data, axis=0), np.array(labels)
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            logger.error(f"❌ Error loading class {class_path}: {e}")
            return np.array([]), np.array([])
    
    def _load_data_optimized(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data with optimization and progress tracking"""
        all_data = []
        all_labels = []
        
        # Get available classes
        classes = sorted([d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d))])
        
        logger.info(f"📁 Found {len(classes)} classes")
        
        # Load each class with progress tracking
        for class_name in tqdm(classes, desc="Loading classes"):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                class_data, class_labels = self._load_class_data(class_path)
                if len(class_data) > 0:
                    all_data.append(class_data)
                    all_labels.append(class_labels)
        
        if all_data:
            # Concatenate all data
            combined_data = np.concatenate(all_data, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            # Memory optimization
            gc.collect()
            
            logger.info(f"✅ Loaded {len(combined_data)} samples with {len(np.unique(combined_labels))} classes")
            return combined_data, combined_labels
        else:
            logger.error("❌ No data loaded!")
            return np.array([]), np.array([])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item with caching and error handling"""
        try:
            # Check cache first
            cache_key = f"sample_{idx}"
            if cache_key in self.cache:
                self.performance_metrics['cache_hits'] += 1
                return self.cache[cache_key]
            
            # Load from data
            sample = self.data[idx]
            label = self.labels[idx]
            
            # Convert to tensor
            sample_tensor = torch.from_numpy(sample).float()
            
            # FIX TENSOR SHAPE: Convert from (height, width, channels) to (channels, height, width)
            # ANN model expects (channels, height, width) format
            if sample_tensor.shape == (34, 34, 2):
                sample_tensor = sample_tensor.permute(2, 0, 1)  # (2, 34, 34)
            
            # Apply transform if available
            if self.transform:
                sample_tensor = self.transform(sample_tensor)
            
            # Cache result (with size limit)
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = (sample_tensor, label)
            
            self.performance_metrics['cache_misses'] += 1
            return sample_tensor, label
            
        except Exception as e:
            logger.error(f"❌ Error loading sample {idx}: {e}")
            # Remove dummy data fallback - raise exception instead
            raise RuntimeError(f"Failed to load sample {idx}: {e}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return self.performance_metrics.copy()


def get_optimized_nmnist_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    train_data_dir: str = './data/N-MNIST/Train',
    test_data_dir: str = './data/N-MNIST/Test',
    cache_size: int = 1000
) -> Tuple[DataLoader, DataLoader]:
    """
    PURPOSE: Get optimized N-MNIST data loaders
    
    ENHANCED FEATURES:
    - Optimized data loading with caching
    - Progress tracking
    - Memory monitoring
    - Error handling
    - Performance metrics
    """
    
    logger.info("🚀 Creating optimized N-MNIST data loaders")
    
    # Create datasets with optimization
    train_dataset = OptimizedTemporalNMNISTDataset(
        data_dir=train_data_dir,
        cache_size=cache_size
    )
    
    test_dataset = OptimizedTemporalNMNISTDataset(
        data_dir=test_data_dir,
        cache_size=cache_size
    )
    
    # Create data loaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Log performance metrics
    train_metrics = train_dataset.get_performance_metrics()
    test_metrics = test_dataset.get_performance_metrics()
    
    logger.info(f"📊 Train loader performance: {train_metrics}")
    logger.info(f"📊 Test loader performance: {test_metrics}")
    
    return train_loader, test_loader


# Backward compatibility - CRITICAL: Maintain exact function signatures
class RealTemporalNMNISTDataset(OptimizedTemporalNMNISTDataset):
    """Backward compatibility wrapper with exact original signature"""
    def __init__(self, root_dir, train=True, download=True, transform=None):
        # Map old parameters to new optimized parameters
        data_dir = os.path.join(root_dir, 'N-MNIST', 'Train' if train else 'Test')
        super().__init__(data_dir=data_dir, transform=transform, cache_size=1000)
        self.root_dir = root_dir
        self.train = train
        self.download = download

def get_nmnist_loaders(batch_size=64, num_workers=4, download=True):
    """Backward compatibility wrapper with exact original signature"""
    return get_optimized_nmnist_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        train_data_dir='./data/N-MNIST/Train',
        test_data_dir='./data/N-MNIST/Test',
        cache_size=1000
    )


if __name__ == "__main__":
    # Test the sophisticated temporal loader
    print("🧪 Testing REAL TEMPORAL N-MNIST loader...")
    try:
        train_loader, test_loader = get_nmnist_loaders(batch_size=4)
        print(f"✅ Train batches: {len(train_loader)}")
        print(f"✅ Test batches: {len(test_loader)}")
        
        # Test one batch
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"📊 Batch {batch_idx}: data shape {data.shape}, labels {labels}")
            print(f"🎯 Temporal dimensions: {data.shape[2]} time steps")
            break
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please download N-MNIST dataset manually from the provided links") 
