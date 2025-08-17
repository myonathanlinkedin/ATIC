"""
Configuration constants for SNN vs ANN benchmark
Centralizes all hardcoded values and magic numbers
"""


# DATASET CONSTANTS
class DatasetConfig:
    """Dataset-specific configuration constants"""
    
    # N-MNIST Dataset
    NMNIST_INPUT_SIZE = 34
    NMNIST_CHANNELS = 2  # ON and OFF events
    NMNIST_NUM_CLASSES = 10
    NMNIST_TIME_STEPS = 100
    NMNIST_SPIKE_THRESHOLD = 0.3
    NMNIST_TEMPORAL_DECAY = 0.1
    NMNIST_NOISE_LEVEL = 0.05
    NMNIST_TEMPORAL_RESOLUTION = 1.0  # ms
    NMNIST_MAX_SPIKE_RATE = 100.0  # Hz
    NMNIST_EVENT_SIZE = 4  # bytes per event
    
    # SHD Dataset
    SHD_INPUT_UNITS = 700
    SHD_NUM_CLASSES = 20
    SHD_MAX_TIME = 1000  # 1 second duration
    SHD_TEMPORAL_KERNELS = [10, 8, 6]
    SHD_TEMPORAL_STRIDES = [2, 2, 2]


# MODEL CONSTANTS
class ModelConfig:
    """Model architecture configuration constants"""
    
    # Default Architecture
    DEFAULT_HIDDEN_DIMS = (32, 64, 128)
    DEFAULT_DROPOUT_RATE = 0.2
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_WEIGHT_DECAY = 1e-4
    DEFAULT_NUM_EPOCHS = 20
    
    # SNN Specific
    DEFAULT_TIME_STEPS = 20
    DEFAULT_DECAY_LAMBDA = 0.05
    DEFAULT_MEMBRANE_THRESHOLD = 1.0
    DEFAULT_MEMBRANE_DECAY = 0.9
    DEFAULT_SYNAPTIC_DECAY = 0.1
    
    # Feature Size Calculations
    CONV_KERNEL_SIZE = 3
    CONV_STRIDE = 2
    CONV_PADDING = 1
    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2
    
    # FC Layer Sizes
    FC_HIDDEN_512 = 512
    FC_HIDDEN_256 = 256


# TRAINING CONSTANTS
class TrainingConfig:
    """Training configuration constants"""
    
    # Batch Processing
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 4
    
    # Monitoring
    METRICS_LOG_INTERVAL = 10
    CHECKPOINT_SAVE_INTERVAL = 5
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Mixed Precision
    USE_MIXED_PRECISION = True
    GRAD_SCALER_ENABLED = True


# HARDWARE CONSTANTS
class HardwareConfig:
    """Hardware and performance configuration constants"""
    
    # GPU Memory Management
    GPU_MEMORY_FRACTION = 0.8
    GPU_MEMORY_CLEAR_INTERVAL = 100
    
    # Energy Monitoring
    ENERGY_MONITORING_INTERVAL = 10
    DEFAULT_CPU_POWER_ESTIMATE = 0.1
    DEFAULT_GPU_POWER_ESTIMATE = 50.0
    
    # Temperature Thresholds
    MAX_GPU_TEMPERATURE = 85.0
    WARNING_GPU_TEMPERATURE = 75.0


# EVALUATION CONSTANTS
class EvaluationConfig:
    """Evaluation and benchmarking constants"""
    
    # Statistical Analysis
    CONFIDENCE_LEVEL = 0.95
    MIN_SAMPLE_SIZE = 30
    
    # Performance Metrics
    MIN_SPIKE_RATE = 0.001
    MAX_TEMPORAL_EFFICIENCY = 1.0
    
    # Comparison Thresholds
    SIGNIFICANT_ACCURACY_DIFFERENCE = 1.0  # percentage points
    SIGNIFICANT_TIME_DIFFERENCE = 0.1  # seconds


# VISUALIZATION CONSTANTS
class VisualizationConfig:
    """Plotting and visualization constants"""
    
    # Figure Sizes
    DEFAULT_FIGURE_SIZE = (12, 8)
    COMPARISON_FIGURE_SIZE = (15, 6)
    SUMMARY_FIGURE_SIZE = (16, 12)
    
    # DPI Settings
    HIGH_DPI = 300
    MEDIUM_DPI = 150
    
    # Color Schemes
    ANN_COLOR = '#ff7f0e'
    SNN_COLOR = '#2ca02c'
    
    # Font Sizes
    TITLE_FONT_SIZE = 16
    AXIS_FONT_SIZE = 14
    LEGEND_FONT_SIZE = 12
    LABEL_FONT_SIZE = 10


# ERROR MESSAGES
class ErrorMessages:
    """Standardized error messages"""
    
    # Dataset Errors
    DATASET_NOT_FOUND = ("Dataset not found. Please download manually "
                         "from the specified URL.")
    INVALID_DATA_FORMAT = ("Invalid data format. Expected binary event data.")
    
    # Model Errors
    INVALID_MODEL_TYPE = "Invalid model type. Must be 'SNN' or 'ANN'."
    MODEL_NOT_LOADED = "Failed to load pre-trained model."
    
    # Hardware Errors
    GPU_NOT_AVAILABLE = ("CUDA GPU not available. Falling back to CPU.")
    MEMORY_ERROR = ("GPU memory insufficient. Try reducing batch size.")
    
    # Training Errors
    TRAINING_FAILED = ("Training failed. Check data and model configuration.")
    CHECKPOINT_CORRUPTED = ("Checkpoint file corrupted. Starting fresh "
                            "training.")


# SUCCESS MESSAGES
class SuccessMessages:
    """Standardized success messages"""
    
    # Training
    TRAINING_COMPLETE = "Training completed successfully!"
    MODEL_SAVED = "Model saved successfully."
    CHECKPOINT_LOADED = "Checkpoint loaded successfully."
    
    # Evaluation
    EVALUATION_COMPLETE = "Evaluation completed successfully!"
    BENCHMARK_COMPLETE = "Benchmark completed successfully!"
    
    # Visualization
    PLOTS_GENERATED = "Plots generated successfully!"
    RESULTS_SAVED = "Results saved successfully!"


# FILE PATHS
class FilePaths:
    """Standardized file path constants"""
    
    # Directories
    RESULTS_DIR = './results'
    LOGS_DIR = './results/logs'
    CHARTS_DIR = './results/charts'
    DATA_DIR = './data'
    
    # Model Files
    SNN_CHECKPOINT = 'best_snn_model.pth'
    ANN_CHECKPOINT = 'best_ann_model.pth'
    SNN_HISTORY = 'snn_training_history.json'
    ANN_HISTORY = 'ann_training_history.json'
    
    # Results Files
    BENCHMARK_RESULTS = 'complete_benchmark_results.json'
    CONFIG_FILE = 'config.json'
    SUMMARY_CSV = 'summary.csv'


# URLS
class URLs:
    """Dataset file references - REAL DATA FILES AVAILABLE LOCALLY"""
    
    # N-MNIST Files (Local data directory)
    NMNIST_MENDELEY_FILES = ["Train.zip", "Test.zip"]  # data/N-MNIST/
    NMNIST_OFFICIAL_FILES = ["Train.zip", "Test.zip"]  # data/N-MNIST/
    
    # SHD Files (Local data directory)
    SHD_FILES = ["shd_train.h5", "shd_test.h5"]  # data/SHD/ 
