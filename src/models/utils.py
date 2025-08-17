import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SophisticatedETADPooling(nn.Module):
    """
    SOPHISTICATED ETAD: Advanced Exponential Time-Weighted Activation Decay
    with Multi-Scale Temporal Processing, Adaptive Decay Rates, and Attention Mechanisms
    """
    
    def __init__(self, decay_lambda: float = None, time_window: int = None,
                 adaptive_decay: bool = True, temporal_attention: bool = True,
                 multi_scale: bool = True, attention_heads: int = 4):
        super().__init__()
        
        # Use constants if not provided
        if decay_lambda is None:
            from src.config.constants import ModelConfig
            decay_lambda = ModelConfig.DEFAULT_DECAY_LAMBDA
        if time_window is None:
            from src.config.constants import DatasetConfig
            time_window = DatasetConfig.NMNIST_TIME_STEPS
        self.decay_lambda = decay_lambda
        self.time_window = time_window
        self.adaptive_decay = adaptive_decay
        self.temporal_attention = temporal_attention
        self.multi_scale = multi_scale
        self.attention_heads = attention_heads
        
        # Register temporal steps buffer
        self.register_buffer('time_steps', 
                           torch.arange(time_window, dtype=torch.float32))
        
        # SOPHISTICATED: Multi-scale temporal processing - ENHANCED DEVICE HANDLING
        if multi_scale:
            self.temporal_scales = [1, 2, 4, 8]  # Multi-scale temporal windows
            self.scale_weights = nn.Parameter(torch.ones(len(self.temporal_scales)))
        
        # SOPHISTICATED: Adaptive decay parameters - ENHANCED DEVICE HANDLING
        if adaptive_decay:
            self.decay_alpha = nn.Parameter(torch.tensor(decay_lambda))
            self.decay_beta = nn.Parameter(torch.tensor(0.1))
            self.decay_gamma = nn.Parameter(torch.tensor(0.05))
            
            # Learnable temporal importance weights - ENHANCED DEVICE HANDLING
            self.temporal_importance = nn.Parameter(torch.ones(time_window))
            
            # Adaptive threshold mechanism - ENHANCED DEVICE HANDLING
            self.adaptive_threshold = nn.Parameter(torch.tensor(0.5))
        
        # SOPHISTICATED: Temporal attention mechanism - ENHANCED DEVICE HANDLING
        if temporal_attention:
            self.temporal_attention = MultiHeadTemporalAttention(
                time_window=time_window,
                num_heads=attention_heads
            )
        
        # SOPHISTICATED: Temporal convolution for feature extraction
        self.temporal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=time_window,
            kernel_size=3,
            padding=1
        )
        
        # SOPHISTICATED: Learnable temporal dynamics - ENHANCED DEVICE HANDLING
        self.temporal_dynamics = nn.Parameter(torch.zeros(time_window))
        
        # SOPHISTICATED: Temporal normalization
        self.temporal_norm = nn.LayerNorm(time_window)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SOPHISTICATED ETAD pooling with advanced temporal processing
        """
        # Handle all tensor shapes robustly
        if len(x.shape) == 3:  # (batch, height, width)
            x = x.unsqueeze(1).unsqueeze(2)  # Add channel and time dimensions
        elif len(x.shape) == 4:  # (batch, channels, height, width)
            x = x.unsqueeze(2)  # Add time dimension
        elif len(x.shape) == 5:  # (batch, channels, time, height, width)
            pass  # Already correct format
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        batch_size, channels, time_steps, height, width = x.shape
        
        # SOPHISTICATED: Multi-scale temporal processing
        if self.multi_scale:
            multi_scale_outputs = []
            
            for scale_idx, scale in enumerate(self.temporal_scales):
                # Downsample temporal dimension
                if scale > 1:
                    scale_time_steps = time_steps // scale
                    x_scaled = F.adaptive_avg_pool3d(
                        x, (scale_time_steps, height, width)
                    )
                else:
                    x_scaled = x
                    scale_time_steps = time_steps
                
                # Process each scale
                scale_output = self._process_temporal_scale(
                    x_scaled, scale_time_steps, height, width
                )
                
                # Upsample back to original temporal resolution
                if scale > 1:
                    # Ensure scale_output has the correct spatial dimensions
                    if scale_output.dim() == 3:  # (batch, channels, time)
                        scale_output = scale_output.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
                    
                    scale_output = F.interpolate(
                        scale_output,
                        size=(time_steps, height, width),
                        mode='trilinear',
                        align_corners=False
                    )
                
                # Ensure all scale outputs have the same spatial dimensions
                if scale_output.dim() == 3:  # (batch, channels, time)
                    scale_output = scale_output.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
                
                # Ensure all scale outputs have the same spatial dimensions
                if scale_output.dim() == 3:  # (batch, channels, time)
                    scale_output = scale_output.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
                
                # Ensure consistent spatial dimensions across all scales
                if scale_output.size(-1) != width or scale_output.size(-2) != height:
                    scale_output = F.interpolate(
                        scale_output,
                        size=(time_steps, height, width),
                        mode='trilinear',
                        align_corners=False
                    )
                
                multi_scale_outputs.append(scale_output)
            
            # Combine multi-scale outputs
            x = torch.stack(multi_scale_outputs, dim=1)
            x = torch.sum(x * self.scale_weights.view(1, -1, 1, 1, 1), dim=1)
        
        # SOPHISTICATED: Adaptive temporal processing
        if self.adaptive_decay:
            x = self._apply_adaptive_temporal_processing(x, time_steps)
        
        # SOPHISTICATED: Temporal attention mechanism
        if self.temporal_attention:
            x = self._apply_temporal_attention(x, time_steps)
        
        # SOPHISTICATED: Learnable temporal dynamics
        # Ensure we don't index beyond the available size
        max_time_steps = min(time_steps, self.temporal_dynamics.size(0))
        temporal_weights = torch.sigmoid(self.temporal_dynamics[:max_time_steps])
        
        # Pad or truncate to match time_steps
        if max_time_steps < time_steps:
            # Pad with zeros if needed
            padding = torch.zeros(time_steps - max_time_steps, device=temporal_weights.device)
            temporal_weights = torch.cat([temporal_weights, padding])
        elif max_time_steps > time_steps:
            # Truncate if needed
            temporal_weights = temporal_weights[:time_steps]
            
        temporal_weights = self.temporal_norm(temporal_weights)
        temporal_weights = temporal_weights.view(1, 1, -1, 1, 1)
        
        # Apply sophisticated temporal pooling
        weighted_spikes = x * temporal_weights
        pooled = torch.sum(weighted_spikes, dim=2)
        
        # Ensure output maintains proper dimensions for FC layers
        if pooled.dim() == 4:  # (batch, channels, height, width)
            # Preserve spatial dimensions for convolutional layers
            # Only flatten at the very end for FC layers
            return pooled
        elif pooled.dim() == 3:  # (batch, channels, features)
            # Already flattened
            return pooled
        elif pooled.dim() == 2:  # (batch, features)
            # Already flattened
            return pooled
        else:
            # Flatten any other shape
            batch_size = pooled.size(0)
            return pooled.view(batch_size, -1)
    
    def _process_temporal_scale(self, x: torch.Tensor, time_steps: int, 
                               height: int, width: int) -> torch.Tensor:
        """Process temporal data at a specific scale"""
        batch_size, channels, time_steps, height, width = x.shape
        
        # SOPHISTICATED: Temporal feature extraction
        # Average over spatial dimensions to get temporal features
        x_temporal = x.mean(dim=(3, 4))  # (batch_size, channels, time_steps)
        x_temporal = x_temporal.view(batch_size * channels, 1, time_steps)  # (batch*channels, 1, time_steps)
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(x_temporal)  # (batch*channels, time_window, time_steps)
        
        # Reshape back
        temporal_features = temporal_features.view(batch_size, channels, 
                                                self.time_window, time_steps)
        temporal_features = temporal_features.mean(dim=2)  # Average over time_window
        temporal_features = temporal_features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        
        return temporal_features
    
    def _apply_adaptive_temporal_processing(self, x: torch.Tensor, 
                                          time_steps: int) -> torch.Tensor:
        """Apply adaptive temporal processing with learnable dynamics"""
        batch_size, channels, time_steps, height, width = x.shape
        
        # SOPHISTICATED: Adaptive decay computation
        # Ensure we don't index beyond the available size
        max_time_steps = min(time_steps, self.time_steps.size(0))
        base_decay = torch.exp(-self.decay_alpha * self.time_steps[:max_time_steps])
        adaptive_factor = torch.sigmoid(self.decay_beta * self.time_steps[:max_time_steps])
        importance_weights = torch.sigmoid(self.temporal_importance[:max_time_steps])
        
        # Pad or truncate to match time_steps
        if max_time_steps < time_steps:
            # Pad with zeros if needed
            padding = torch.zeros(time_steps - max_time_steps, device=base_decay.device)
            base_decay = torch.cat([base_decay, padding])
            adaptive_factor = torch.cat([adaptive_factor, padding])
            importance_weights = torch.cat([importance_weights, padding])
        elif max_time_steps > time_steps:
            # Truncate if needed
            base_decay = base_decay[:time_steps]
            adaptive_factor = adaptive_factor[:time_steps]
            importance_weights = importance_weights[:time_steps]
        
        # Combine adaptive factors
        adaptive_decay = base_decay * adaptive_factor * importance_weights
        
        # SOPHISTICATED: Threshold-based temporal filtering
        temporal_threshold = torch.sigmoid(self.adaptive_threshold)
        temporal_mask = (adaptive_decay > temporal_threshold).long().float()
        
        # Apply adaptive processing
        adaptive_weights = adaptive_decay * temporal_mask
        adaptive_weights = adaptive_weights.view(1, 1, -1, 1, 1)
        
        return x * adaptive_weights
    
    def _apply_temporal_attention(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Apply sophisticated temporal attention mechanism"""
        return self.temporal_attention(x, time_steps)


class MultiHeadTemporalAttention(nn.Module):
    """
    SOPHISTICATED: Multi-head temporal attention mechanism
    """
    
    def __init__(self, time_window: int, num_heads: int = 4):
        super().__init__()
        self.time_window = time_window
        self.num_heads = min(num_heads, time_window)  # Ensure num_heads <= time_window
        self.head_dim = max(1, time_window // self.num_heads)  # Ensure head_dim >= 1
        
        # SOPHISTICATED: Multi-head attention parameters - ENHANCED DEVICE HANDLING
        # Initialize with zeros but will be moved to correct device when model is moved
        self.query = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.key = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.value = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        
        # SOPHISTICATED: Temporal position encoding - ENHANCED DEVICE HANDLING
        self.temporal_pos_encoding = nn.Parameter(torch.zeros(time_window))
        
        # SOPHISTICATED: Output projection
        self.output_projection = nn.Linear(time_window, time_window)
    
    def forward(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        Apply multi-head temporal attention
        """
        batch_size, channels, time_steps, height, width = x.shape
        
        # SOPHISTICATED: Compute temporal features
        x_temporal = x.mean(dim=(3, 4))  # Average over spatial dimensions
        x_temporal = x_temporal.view(batch_size * channels, time_steps)
        
        # SOPHISTICATED: Add temporal position encoding
        # Ensure we don't index beyond the available size
        try:
            max_time_steps = min(time_steps, self.temporal_pos_encoding.size(0))
            pos_encoding = self.temporal_pos_encoding[:max_time_steps]
            
            # Pad or truncate to match time_steps
            if max_time_steps < time_steps:
                # Pad with zeros if needed
                padding = torch.zeros(time_steps - max_time_steps, device=pos_encoding.device)
                pos_encoding = torch.cat([pos_encoding, padding])
            elif max_time_steps > time_steps:
                # Truncate if needed
                pos_encoding = pos_encoding[:time_steps]
        except Exception:
            # Fallback to zeros if indexing fails
            pos_encoding = torch.zeros(time_steps, device=x.device)
            
        x_temporal = x_temporal + pos_encoding.unsqueeze(0)
        
        # SOPHISTICATED: Multi-head attention computation
        attention_outputs = []
        
        for head in range(self.num_heads):
            # Compute attention scores
            query = self.query[head].unsqueeze(0)  # (1, head_dim)
            key = self.key[head].unsqueeze(0)      # (1, head_dim)
            value = self.value[head].unsqueeze(0)   # (1, head_dim)
            
            # Reshape for attention computation with bounds checking
            start_idx = head * self.head_dim
            end_idx = min((head + 1) * self.head_dim, x_temporal.size(1))
            x_head = x_temporal[:, start_idx:end_idx]
            
            # Pad if necessary
            if x_head.size(1) < self.head_dim:
                padding = torch.zeros(x_temporal.size(0), self.head_dim - x_head.size(1), device=x_head.device)
                x_head = torch.cat([x_head, padding], dim=1)
            
            # Compute attention
            attention_scores = torch.matmul(x_head, key.transpose(-2, -1))
            attention_scores = F.softmax(attention_scores, dim=-1)
            
            # Apply attention
            head_output = torch.matmul(attention_scores, value)
            attention_outputs.append(head_output)
        
        # SOPHISTICATED: Combine attention heads
        attention_output = torch.cat(attention_outputs, dim=-1)
        attention_output = self.output_projection(attention_output)
        
        # SOPHISTICATED: Reshape and apply to original tensor
        attention_output = attention_output.view(batch_size, channels, time_steps)
        attention_output = attention_output.unsqueeze(-1).unsqueeze(-1)
        
        return x * attention_output


class ETADPooling(nn.Module):
    """
    LEGACY: Basic ETAD pooling for backward compatibility
    """
    
    def __init__(self, decay_lambda: float = None, time_window: int = None):
        super().__init__()
        
        # Use constants if not provided
        if decay_lambda is None:
            from src.config.constants import ModelConfig
            decay_lambda = ModelConfig.DEFAULT_DECAY_LAMBDA
        if time_window is None:
            from src.config.constants import DatasetConfig
            time_window = DatasetConfig.NMNIST_TIME_STEPS
        super().__init__()
        self.decay_lambda = decay_lambda
        self.time_window = time_window
        self.register_buffer('time_steps', 
                           torch.arange(time_window, dtype=torch.float32))
        self.decay_weights = torch.exp(-self.decay_lambda * self.time_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward pass"""
        if len(x.shape) == 4:
            x = x.unsqueeze(2)
        
        batch_size, channels, time_steps, height, width = x.shape
        
        actual_time_steps = min(time_steps, self.time_window)
        decay_weights = self.decay_weights[:actual_time_steps].view(1, 1, -1, 1, 1)
        decay_weights = decay_weights.to(x.device)
        
        x_truncated = x[:, :, :actual_time_steps, :, :]
        weighted_spikes = x_truncated * decay_weights
        pooled = torch.sum(weighted_spikes, dim=2)
        
        return pooled


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_neurons(model: nn.Module, input_data: torch.Tensor) -> Dict[str, int]:
    """Count active neurons in each layer during forward pass"""
    active_counts = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Count non-zero activations as active neurons
                active_neurons = torch.count_nonzero(output).item()
                active_counts[name] = active_neurons
        return hook
    
    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return active_counts


def calculate_energy_efficiency(
    model: nn.Module, 
    input_data: torch.Tensor,
    power_consumption: float,
    inference_time: float
) -> Dict[str, float]:
    """Calculate energy efficiency metrics"""
    total_params = count_parameters(model)
    active_neurons = count_active_neurons(model, input_data)
    
    # Energy per inference (Joules)
    energy_per_inference = power_consumption * inference_time / 1000  # Convert to Joules
    
    # Energy efficiency metrics
    energy_per_param = energy_per_inference / total_params if total_params > 0 else 0
    active_neurons_sum = sum(active_neurons.values())
    energy_per_active_neuron = (energy_per_inference / active_neurons_sum 
                               if active_neurons_sum > 0 else 0)
    
    return {
        'energy_per_inference': energy_per_inference,
        'energy_per_param': energy_per_param,
        'energy_per_active_neuron': energy_per_active_neuron,
        'total_params': total_params,
        'active_neurons': active_neurons
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Create comprehensive model summary"""
    # Count parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate model size
    model_size_mb = get_model_size_mb(model)
    
    # Create summary
    summary = {
        'model_type': model.__class__.__name__,
        'input_shape': input_shape,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': model_size_mb,
        'architecture': str(model)
    }
    
    return summary 
