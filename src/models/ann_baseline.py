import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .utils import count_parameters, create_model_summary

class ANNBaseline(nn.Module):
    """
    PURPOSE: Baseline Artificial Neural Network for comparison with SNN
    
    ENHANCED FEATURES:
    - Proper regularization to prevent overfitting
    - Dropout layers for generalization
    - Batch normalization for stable training
    - Weight decay for regularization
    """
    
    def __init__(self, input_channels: int = 2, num_classes: int = 10, 
                 hidden_dims: Tuple[int, ...] = (32, 64, 128), dropout_rate: float = 0.5):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # FIXED: Optimal 3-Conv + 3-FC architecture with balanced regularization
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout2d(0.2),  # FIXED: Moderate spatial dropout
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout2d(0.2),  # FIXED: Moderate spatial dropout
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.ReLU(),
                nn.Dropout2d(0.2),  # FIXED: Moderate spatial dropout
                nn.MaxPool2d(2)
            )
        ])
        
        # Calculate feature size for FC layers
        feature_size = hidden_dims[2] * 4 * 4  # After 3 max pooling layers
        
        # FIXED: Balanced FC layers with optimal regularization
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # FIXED: Optimal dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # FIXED: Optimal dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),  # FIXED: Moderate dropout
            nn.Linear(128, num_classes)
        )
        
        # FIXED: Initialize weights properly for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with proper regularization
        
        ENHANCED FEATURES:
        - Dropout during training for regularization
        - Batch normalization for stable gradients
        - Proper feature extraction
        """
        # Handle temporal input by averaging over time dimension
        if len(x.shape) == 5:  # (batch, channels, time, height, width)
            x = x.mean(dim=2)  # Average over time dimension
        
        # Apply convolutional layers with regularization
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Apply fully connected layers with regularization
        # BatchNorm1d requires batch_size >= 2 in training; guard by eval on tiny batch
        if self.training and x.size(0) < 2:
            prev = self.training
            try:
                self.eval()
                x = self.fc_layers(x)
            finally:
                self.train(prev)
        else:
            x = self.fc_layers(x)
        
        return x
    
    def get_model_info(self, input_shape: Tuple[int, ...]) -> dict:
        """Get model information and summary"""
        return create_model_summary(self, input_shape)

class ANNBaselineSHD(nn.Module):
    """
    ANN baseline specifically for SHD dataset
    Handles temporal spike data differently
    """
    
    def __init__(
        self,
        input_units: int = 700,
        num_classes: int = 20,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = None
    ):
        # Use constant if no dropout_rate provided
        if dropout_rate is None:
            from src.config.constants import ModelConfig
            dropout_rate = ModelConfig.DEFAULT_DROPOUT_RATE  # Use constant instead of hardcoded value
        super().__init__()
        
        self.input_units = input_units
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_units, 256, kernel_size=10, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(256, 128, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout_rate)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SHD data
        Args:
            x: Input tensor of shape (batch_size, input_units, time_steps)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Apply temporal convolutions
        x = self.temporal_conv(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def get_model_info(self, input_shape: Tuple[int, ...]) -> dict:
        """Get model information and summary"""
        return create_model_summary(self, input_shape) 
