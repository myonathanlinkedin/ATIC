import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class SHDDataset(Dataset):
    """SHD (Spiking Heidelberg Digits) Dataset Loader"""
    
    def __init__(self, root: str, train: bool = True, 
                 transform=None, download: bool = True):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(root, 'SHD'), exist_ok=True)
        
        # Load data
        self.data, self.targets = self._load_data()
    
    def _download(self):
        """Download SHD dataset if not present"""
        # Check if files already exist
        split = 'train' if self.train else 'test'
        filepath = os.path.join(self.root, 'SHD', f'shd_{split}.h5')
        
        if os.path.exists(filepath):
            return
        
        # Handle gzipped files
        gz_filepath = filepath + '.gz'
        if os.path.exists(gz_filepath):
            import gzip
            import shutil
            print(f"Extracting {gz_filepath}...")
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SHD dataset from HDF5 file"""
        split = 'train' if self.train else 'test'
        filepath = os.path.join(self.root, 'SHD', f'shd_{split}.h5')
        
        # Download if needed
        if self.download:
            self._download()
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"SHD {split} file not found: {filepath}"
            )
        
        print(f"Loading SHD data from {filepath}...")
        
        with h5py.File(filepath, 'r') as f:
            # Load spike times and indices
            spike_times = f['spikes']['times'][:]
            spike_units = f['spikes']['units'][:]
            labels = f['labels'][:]
            
            print(f"Loaded {len(labels)} samples, "
                  f"{len(spike_times)} total spikes")
            print(f"Spike times shape: {spike_times.shape}, "
                  f"dtype: {spike_times.dtype}")
            print(f"Spike units shape: {spike_units.shape}, "
                  f"dtype: {spike_units.dtype}")
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            
            # Convert to spike trains
            from src.config.constants import DatasetConfig
            max_time = DatasetConfig.SHD_MAX_TIME
            num_units = DatasetConfig.SHD_INPUT_UNITS
            
            data = []
            processed_labels = []
            
            # SHD data structure: each sample has its own list of spike times and units
            # spike_times[i] contains the spike times for sample i
            # spike_units[i] contains the spike units for sample i
            
            print(f"Processing {len(labels)} samples...")
            
            # Process each sample
            for i in range(len(labels)):
                # Create empty spike train for this sample
                spike_train = np.zeros((num_units, max_time), 
                                      dtype=np.float32)
                
                # Get spikes for this sample
                sample_times = spike_times[i]
                sample_units = spike_units[i]
                
                # Convert to numpy arrays if they're not already
                if (hasattr(sample_times, '__len__') and 
                    len(sample_times) > 0):
                    sample_times = np.array(sample_times)
                    sample_units = np.array(sample_units)
                    
                    # Add spikes to spike train
                    for spike_time, unit in zip(sample_times, sample_units):
                        if (isinstance(unit, (int, np.integer)) and 
                            isinstance(spike_time, 
                                     (int, float, np.number))):
                            if unit < num_units and spike_time < max_time:
                                spike_train[unit, int(spike_time)] = 1.0
                
                data.append(spike_train)
                processed_labels.append(labels[i])
            
            print(f"✅ Successfully processed {len(data)} SHD samples")
            return np.array(data), np.array(processed_labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spike_data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            spike_data = self.transform(spike_data)
        
        return torch.FloatTensor(spike_data), target


def get_shd_loaders(
    root: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Get SHD train and test dataloaders"""
    
    train_dataset = SHDDataset(root=root, train=True, download=download)
    test_dataset = SHDDataset(root=root, train=False, download=download)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 
