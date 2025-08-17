"""
Training modules for SNN and ANN models
"""

from .train_ann import ANNTrainer, train_ann_model
from .train_snn import SNNTrainer, train_snn_model

__all__ = [
    'ANNTrainer',
    'train_ann_model',
    'SNNTrainer', 
    'train_snn_model'
] 
