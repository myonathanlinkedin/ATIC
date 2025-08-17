#!/usr/bin/env python3
"""
Enhanced Cognitive Neuroscience Framework: SNN vs ANN Benchmark with Comprehensive Neuromorphic Assessment

PURPOSE: Main entry point for comprehensive neuromorphic benchmarking with cognitive neuroscience focus
- Implements enhanced SNN with ATIC, NAA, and brain region mapping
- Provides comprehensive evaluation using CNAF with cognitive neuroscience metrics
- Generates professional benchmark results suitable for research publication
- Supports skip training and flexible configuration

ENHANCED COGNITIVE NEUROSCIENCE FRAMEWORK:
- Adaptive Temporal Information Compression (ATIC): Information-theoretic optimal temporal processing
- Neural Architecture Adaptation (NAA): Real-time architecture adaptation based on input characteristics
- Comprehensive Neuromorphic Assessment Framework (CNAF): Multi-dimensional evaluation with cognitive focus
- Brain Region Mapping: V1 (edge detection), V2 (shape processing), V4 (color/form), IT (object recognition)
- Cognitive Process Analysis: Attention mechanisms, memory processes, executive functions
- Theoretical Neuroscience Framework: Temporal binding, predictive coding, neural synchronization
- Enhanced Metrics: BPI (Biological Plausibility Index), TEI (Temporal Efficiency Index), NPI (Neuromorphic Performance Index)

USAGE:
    python main.py [options]

EXAMPLES:
    # Full benchmark with enhanced cognitive neuroscience framework
    python main.py --epochs 20 --batch-size 64
    
    # Skip training if models exist
    python main.py --skip-training
    
    # Skip specific model training
    python main.py --skip-snn --skip-ann
    
    # Plot only from existing results
    python main.py --plot-only
"""

import argparse
import os
import sys
import json
import torch
from datetime import datetime
import numpy as np

from src.evaluation.benchmark_runner import run_benchmark
from src.logging.plot_benchmark import create_benchmark_plots
from src.logging.save_metrics import log_benchmark_results
from src.models.utils import set_seed


def main():
    # CRITICAL FIX: Set fixed random seed for reproducibility
    set_seed(42)
    
    # CRITICAL FIX: Memory optimization for RTX 4090 Mobile - CONDITIONAL
    # Only set if CUDA is available and platform supports it
    if torch.cuda.is_available():
        try:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        except Exception:
            # Silently ignore if not supported on this platform
            pass
    """
    PURPOSE: Main function for enhanced SNN vs ANN benchmark with cognitive neuroscience focus
    
    EXECUTION FLOW:
    1. Parse command-line arguments
    2. Validate configuration and device availability
    3. Create output directories
    4. Save configuration for reproducibility
    5. Run comprehensive benchmark with enhanced cognitive neuroscience metrics
    6. Generate professional plots and reports suitable for research publication
    7. Handle errors gracefully
    
    COMMAND-LINE ARGUMENTS:
    - device: Target device for computation ('cuda' or 'cpu')
    - batch-size: Training batch size (default: 64)
    - epochs: Number of training epochs (default: 20)
    - learning-rate: Learning rate for optimization (default: 1e-3)
    - save-dir: Directory to save results (default: './results')
    - datasets: List of datasets to use (default: ['nmnist'])
    - skip-training: Skip training if models exist (default: False)
    - skip-snn: Skip SNN training specifically (default: False)
    - skip-ann: Skip ANN training specifically (default: False)
    - plot-only: Only generate plots from existing results (default: False)
    
    ENHANCED COGNITIVE NEUROSCIENCE FEATURES:
    - ATIC: Adaptive Temporal Information Compression with information-theoretic optimization
    - NAA: Neural Architecture Adaptation for real-time architecture optimization
    - CNAF: Comprehensive Neuromorphic Assessment Framework with cognitive neuroscience focus
    - Brain Region Mapping: V1, V2, V4, IT cortical mapping for cognitive analysis
    - Cognitive Process Analysis: Attention, memory, executive function assessment
    - Theoretical Neuroscience Framework: Temporal binding, predictive coding, neural synchronization
    - Enhanced Metrics: BPI, TEI, NPI for comprehensive neuromorphic evaluation
    - Professional plotting and reporting suitable for research publication
    
    EXPECTED OUTPUT:
    - Complete benchmark results with enhanced cognitive neuroscience metrics
    - Professional plots suitable for research publication
    - Comprehensive reports with brain region analysis and cognitive process evaluation
    - Model checkpoints and training history with enhanced framework data
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Neuromorphic Framework: SNN vs ANN Benchmark with Comprehensive Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full benchmark with enhanced cognitive neuroscience framework
  python main.py --epochs 20 --batch-size 64
  
  # Skip training if models exist
  python main.py --skip-training
  
  # Skip specific model training
  python main.py --skip-snn --skip-ann
  
  # Plot only from existing results
  python main.py --plot-only

ENHANCED FRAMEWORK COMPONENTS:
  - ATIC: Adaptive Temporal Information Compression
  - NAA: Neural Architecture Adaptation
  - CNAF: Comprehensive Neuromorphic Assessment Framework
  - Brain Region Mapping: V1, V2, V4, IT cortical mapping
  - Cognitive Process Analysis: Attention, memory, executive functions
  - Theoretical Framework: Temporal Binding, Predictive Coding, Neural Synchronization
  - Enhanced Metrics: BPI, TEI, NPI for comprehensive evaluation
        """
    )
    
    # Add arguments
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation (cuda/cpu)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Training batch size (default: 32 for better gradient estimates)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20,
        help='Number of training epochs (default: 20 for laptop safety)'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=1e-3,
        help='Learning rate for optimization'
    )
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='./results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        default=['nmnist', 'shd'],
        choices=['nmnist', 'shd'],
        help='List of datasets to use (default: nmnist shd)'
    )
    parser.add_argument(
        '--skip-training', 
        action='store_true',
        help='Skip training if models exist'
    )
    parser.add_argument(
        '--skip-snn', 
        action='store_true',
        help='Skip SNN training specifically'
    )
    parser.add_argument(
        '--skip-ann', 
        action='store_true',
        help='Skip ANN training specifically'
    )
    parser.add_argument(
        '--plot-only', 
        action='store_true',
        help='Only generate plots from existing results'
    )
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'charts'), exist_ok=True)
    
    # Save configuration
    config = {
        'device': args.device,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'datasets': args.datasets,
        'skip_training': args.skip_training,
        'skip_snn': args.skip_snn,
        'skip_ann': args.skip_ann,
        'plot_only': args.plot_only,
        'timestamp': datetime.now().isoformat(),
        'enhanced_features': {
            'atic': 'Adaptive Temporal Information Compression',
            'naa': 'Neural Architecture Adaptation',
            'cnaf': 'Comprehensive Neuromorphic Assessment Framework',
            'brain_mapping': 'Brain Region Mapping (V1, V2, V4, IT)',
            'cognitive_analysis': 'Cognitive Process Analysis',
            'theoretical_framework': 'Theoretical Neuroscience Framework',
            'enhanced_metrics': 'BPI, TEI, NPI'
        }
    }
    
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🚀 Enhanced Neuromorphic Framework: SNN vs ANN Benchmark")
    print(f"📊 Configuration saved to: {config_path}")
    print(f"🎯 Target device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"⏱️  Epochs: {args.epochs}")
    print(f"📚 Datasets: {args.datasets}")
    
    if args.skip_training:
        print("⏭️  Skipping training (models exist)")
    if args.skip_snn:
        print("⏭️  Skipping SNN training")
    if args.skip_ann:
        print("⏭️  Skipping ANN training")
    if args.plot_only:
        print("📊 Plot-only mode")
    
    print("\n🧠 Enhanced Framework Components:")
    print("  - ATIC: Adaptive Temporal Information Compression")
    print("  - NAA: Neural Architecture Adaptation")
    print("  - CNAF: Comprehensive Neuromorphic Assessment Framework")
    print("  - Brain Region Mapping: V1, V2, V4, IT")
    print("  - Cognitive Process Analysis: Attention, Memory, Executive")
    print("  - Theoretical Framework: Temporal Binding, Predictive Coding, Neural Synchronization")
    print("  - Enhanced Metrics: BPI, TEI, NPI")
    
    try:
        if args.plot_only:
            # Generate plots from existing results
            print("\n📊 Generating plots from existing results...")
            
            # CRITICAL FIX: Look for dataset-specific results files
            found_results = False
            for dataset in ['nmnist', 'shd']:
                results_file = os.path.join(args.save_dir, f'complete_benchmark_results_{dataset}.json')
                if os.path.exists(results_file):
                    print(f"✅ Found results for {dataset.upper()}")
                    create_benchmark_plots(results_file, args.save_dir, filename_suffix=f'_{dataset}')
                    found_results = True
            
            if found_results:
                print("✅ Plots generated successfully from existing results!")
            else:
                print("❌ No existing results found. Run benchmark first.")
                print("🔍 Looking for files:")
                print(f"   - {os.path.join(args.save_dir, 'complete_benchmark_results_nmnist.json')}")
                print(f"   - {os.path.join(args.save_dir, 'complete_benchmark_results_shd.json')}")
                return
        else:
            # Run comprehensive benchmark
            print("\n🔬 Running enhanced neuromorphic benchmark...")

            def convert_numpy_to_json(obj):
                """Convert numpy arrays to JSON-serializable format"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_json(item) for item in obj]
                else:
                    return obj
            
            datasets_to_run = args.datasets if isinstance(args.datasets, list) else [args.datasets]
            ran_multiple = len(datasets_to_run) > 1

            for ds in datasets_to_run:
                print(f"\n🔁 Running dataset: {ds}")
                results = run_benchmark(
                    device=args.device,
                    batch_size=args.batch_size,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    save_dir=args.save_dir,
                    skip_training=args.skip_training,
                    skip_snn=args.skip_snn,
                    skip_ann=args.skip_ann,
                    datasets=[ds]
                )

                # Save results (dataset-specific file)
                results_file = os.path.join(args.save_dir, f'complete_benchmark_results_{ds}.json')
                json_results = convert_numpy_to_json(results)
                with open(results_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                
                # Generate plots
                if str(ds).lower() == 'nmnist':
                    print("\n📊 Generating professional plots (N-MNIST main 34 charts)...")
                    print(f"🔧 DEBUG: Results type before plotting: {type(results)}")
                    if isinstance(results, dict):
                        print(f"🔧 DEBUG: Results keys before plotting: {list(results.keys())}")
                    create_benchmark_plots(results_file, args.save_dir, results_data=results, filename_suffix='_nmnist')
                    print("🔧 DEBUG: Plotting completed successfully")
                else:
                    print("\n📊 Generating main charts for SHD with suffix...")
                    create_benchmark_plots(results_file, args.save_dir, results_data=results, filename_suffix='_shd')
                
                # Log results
                print("\n📝 Logging benchmark results...")
                print("🔧 DEBUG: About to call log_benchmark_results")
                log_benchmark_results(results_file, args.save_dir)
                print("🔧 DEBUG: log_benchmark_results completed")

            print("\n✅ Enhanced neuromorphic benchmark completed successfully!")
            print(f"📁 Results saved to: {args.save_dir}")
            print(f"📊 Plots generated in: {os.path.join(args.save_dir, 'charts')}")
            print(f"📝 Logs saved in: {os.path.join(args.save_dir, 'logs')}")
            
    except Exception as e:
        print(f"\n❌ Error during benchmark execution: {str(e)}")
        print("🔍 Check logs for detailed error information")
        import traceback
        print("\n🔧 DETAILED ERROR TRACEBACK:")
        traceback.print_exc()
        print(f"\n🔧 ERROR TYPE: {type(e).__name__}")
        print(f"🔧 ERROR LOCATION: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 