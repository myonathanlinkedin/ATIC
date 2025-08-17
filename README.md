# Enhanced Neuromorphic Benchmark: SNN vs ANN with ATIC and NAA

An end-to-end benchmark for Spiking Neural Networks (SNNs) vs. Artificial Neural Networks (ANNs) on real neuromorphic datasets, featuring advanced temporal processing frameworks, comprehensive neuromorphic metrics, energy monitoring, and publication-ready plots.

## Highlights
- **ATIC (Adaptive Temporal Information Compression)**: Information-theoretic optimal temporal processing with adaptive compression
- **NAA (Neural Architecture Adaptation)**: Real-time architecture optimization based on input characteristics
- Spiking model with Norse-style LIF dynamics and enhanced cognitive neuroscience framework
- ANN baseline with matched capacity
- Combined neuromorphic datasets: N-MNIST + SHD
  - Unified 30-class setting: 0–9 (N-MNIST), 10–29 (SHD)
- Comprehensive metrics (CNAF):
  - BPI (Biological Plausibility Index)
  - TEI (Temporal Efficiency Index)
  - NPI (Neuromorphic Performance Index)
  - Combined comprehensive_score = 0.4×BPI + 0.3×TEI + 0.3×NPI (0–1; you may report ×100)
- Energy monitoring for power/efficiency trends and plots
- Robust logging, charts, and JSON results

## Project Structure

```
snn_project/
  ├─ main.py                          # Entry point (CLI)
  ├─ README.md                        # This guide
  ├─ data/                            # Place datasets here
  │   ├─ N-MNIST/                     # N-MNIST (Train/Test with 0..9)
  │   └─ SHD/                         # SHD (shd_train.h5, shd_test.h5)
  ├─ results/                         # Outputs (JSON, charts, logs)
  │   ├─ complete_benchmark_results.json
  │   ├─ charts/                      # Auto-generated figures
  │   └─ logs/                        # Checkpoints & detailed logs
  └─ src/
      ├─ evaluation/
      │   └─ benchmark_runner.py      # Training, evaluation, metrics, logging
      ├─ models/                      # SNN & ANN models
      │   ├─ snn_etad_improved.py    # Main SNN with ATIC, NAA, and LIF neurons
      │   ├─ ann_baseline.py         # ANN baseline for comparison
      │   └─ utils.py                 # Utility functions and components
      ├─ dataloaders/                 # Dataset loaders
      │   ├─ nmnist_loader.py
      │   └─ shd_loader.py
      ├─ config/                      # Configuration constants
      │   └─ constants.py             # Model, dataset, and training parameters
      └─ logging/                     # Plotting & reporting utilities
```

## Core Frameworks

### ATIC (Adaptive Temporal Information Compression)
The ATIC framework provides information-theoretic optimal temporal processing with:
- **Adaptive Compression**: Dynamic compression based on input complexity
- **Information Entropy**: Computes entropy for optimal temporal processing
- **Temporal Binding**: Biological plausibility through temporal binding mechanisms
- **Real-time Adaptation**: Continuous adaptation to input characteristics

### NAA (Neural Architecture Adaptation)
The NAA framework enables real-time architecture optimization:
- **Complexity Assessment**: Analyzes input complexity in real-time
- **Resource Optimization**: Optimizes computational pathways
- **Adaptive Layer Configuration**: Dynamically selects optimal architecture
- **Cognitive Task Optimization**: Adapts to specific cognitive requirements

## Requirements
- Python 3.10+
- PyTorch (CUDA optional but recommended)
- Common scientific Python packages: numpy, tqdm, h5py, psutil, matplotlib, seaborn

Install PyTorch per your CUDA version from the official website, then:

```bash
pip install numpy tqdm h5py psutil matplotlib seaborn
```

## Datasets
Download from the official sources and place under `snn_project/data`.

- N-MNIST (Neuromorphic MNIST)
  - Source: Mendeley Data — URL: https://data.mendeley.com/datasets/468j46mzdv/1 (DOI: 10.17632/468j46mzdv.1)
  - After download/unpack, expected layout:
    - `snn_project/data/N-MNIST/Train/0..9/...`
    - `snn_project/data/N-MNIST/Test/0..9/...`

- SHD (Spiking Heidelberg Digits)
  - Source: Zenke Lab Datasets — URL: https://zenkelab.org/datasets/
  - Reference: Cramer B., Stradmann Y., Schemmel J., Zenke F. (2020) “The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks”, IEEE TNNLS 31(7)
  - Files needed:
    - `snn_project/data/SHD/shd_train.h5`
    - `snn_project/data/SHD/shd_test.h5`
    - Alternatively place `shd_train.h5.gz` and `shd_test.h5.gz` (the loader will extract)

Notes:
- The benchmark supports using either dataset alone or both combined (default). In combined mode, labels are mapped to a 30-class space (0–9 N‑MNIST, 10–29 SHD).
 - Internally, SHD (700×time) is converted into a 2×34×34 representation for the combined pipeline.

## Quickstart

The benchmark automatically uses ATIC and NAA frameworks for enhanced temporal processing and architecture adaptation.

From the repository root (recommended):

```powershell
# Windows PowerShell, CUDA if available (default run)
$env:Q1_SEEDS="10"; python -m snn_project.main --device cuda --epochs 20 --batch-size 1024 --save-dir snn_project/results

# CPU fallback
$env:Q1_SEEDS="10"; python -m snn_project.main --device cpu --epochs 20 --batch-size 512 --save-dir snn_project/results
```

From inside `snn_project/`:

```powershell
$env:Q1_SEEDS="10"; python main.py --device cuda --epochs 20 --batch-size 1024 --save-dir results
```

Tips:
- If you see out-of-memory or high thermals, reduce `--batch-size` (e.g., 32). 
- Increase `Q1_SEEDS` (e.g., 10) to try multiple random seeds and keep the best empirical result.

### Quick run (single seed) — recommended defaults (epochs 30)

If you want a quick single-seed run with a safe batch on laptops:

```powershell
# From inside snn_project/
$env:Q1_SEEDS="1"; python main.py --device cuda --epochs 30 --batch-size 1024 --save-dir results --datasets nmnist

# Or from repo root:
$env:Q1_SEEDS="1"; python -m snn_project.main --device cuda --epochs 30 --batch-size 1024 --save-dir snn_project/results --datasets nmnist
```

Notes:
- RTX 4090 Mobile 16GB: 1024 is a stable default for both N‑MNIST and SHD. Increase only if VRAM/thermals allow.
- If OOM, step down batch (1024→768→512) and close other GPU processes.

### Per-dataset runs (control batch per dataset)

```powershell
# N-MNIST at batch 1024 (default)
$env:Q1_SEEDS="1"; python main.py --device cuda --epochs 30 --batch-size 1024 --datasets nmnist --save-dir results

# SHD at batch 1024 (default)
$env:Q1_SEEDS="1"; python main.py --device cuda --epochs 30 --batch-size 1024 --datasets shd --save-dir results
```
- With a single seed (`Q1_SEEDS=1`) results vary slightly; increase seeds if needed.

## Outputs
- JSON results: `snn_project/results/complete_benchmark_results.json`
  - Keys of interest:
    - `results.snn.metrics.enhanced_metrics`: `biological_plausibility`, `temporal_efficiency`, `neuromorphic_performance`, `comprehensive_score`
    - `results.snn.atic_metrics`: ATIC framework performance metrics
    - `results.snn.naa_metrics`: NAA framework adaptation metrics
    - The ANN block mirrors the SNN block
- Charts: `snn_project/results/charts/`
  - ATIC sensitivity analysis and temporal binding visualizations
  - NAA architecture adaptation patterns
  - Traditional performance and energy metrics
- Logs + checkpoints: `snn_project/results/logs/`

### Comprehensive Score
Stored in JSON as 0–1:

```
comprehensive_score = 0.4 * biological_plausibility
                    + 0.3 * temporal_efficiency
                    + 0.3 * neuromorphic_performance
```

You can report it as 0–100 by multiplying by 100.

## Evaluate Saved Models (no training)
You can evaluate previously saved models using `--skip-training`. Ensure `--save-dir` points to the same directory where checkpoints were written (default shown below).

```powershell
$env:Q1_SEEDS="1"; python -m snn_project.main --device cuda --epochs 1 --batch-size 512 --skip-training --save-dir snn_project/results
```

Notes:
- Checkpoints are stored under `snn_project/results/logs/` by the default commands above.
- Skip-training loads the model weights and evaluates on the test set (no training resume).

## Custom commands
- Smaller batch if VRAM is limited (inside repo root):
```
$env:Q1_SEEDS="10"; python -m snn_project.main --device cuda --epochs 20 --batch-size 512 --save-dir snn_project/results
```
- CPU-only quick check:
```
$env:Q1_SEEDS="1"; python -m snn_project.main --device cpu --epochs 1 --batch-size 64 --save-dir snn_project/results
```

## Impact of batch=512, epochs=20
- Throughput: higher; fewer steps per epoch.
- Memory: much higher VRAM; may cause OOM on some GPUs. Reduce to 128/256 if needed.
- Stability: BatchNorm more stable with large batches; gradients smoother.
- Convergence: with only 20 epochs, large batches can slightly slow convergence per sample; compensate by more epochs if accuracy plateaus.

## Enhanced Cognitive Neuroscience Framework

The benchmark includes a comprehensive cognitive neuroscience framework:

- **Brain Region Mapping**: Maps SNN layers to visual cortex regions (V1, V2, V4, IT)
- **Cognitive Process Analysis**: Attention mechanisms, memory processes, executive functions
- **Theoretical Neuroscience Validation**: Temporal binding hypothesis, predictive coding theory, neural synchronization
- **Statistical Validation**: Confidence intervals, effect sizes, normality tests, outlier detection

## Advanced Usage
- Multi-seed selection: set `Q1_SEEDS` to run multiple seeds and internally select the best empirical score (JSON schema stays the same).
- Energy monitoring: detailed JSON logs (`results/logs/`) feed power/efficiency charts; you can delete the detailed hardware logs if not needed for analysis.
- Framework customization: Modify ATIC and NAA parameters in `src/config/constants.py`

## Contributing
Pull requests are welcome. Please ensure code clarity and avoid breaking the JSON result schema. For substantial changes, share a short rationale in the PR description.

### Framework Extensions
- **ATIC Extensions**: Add new temporal processing mechanisms or entropy calculations
- **NAA Extensions**: Implement new architecture adaptation strategies
- **Cognitive Framework**: Extend brain region mapping or cognitive process analysis
- **Metrics**: Add new neuromorphic or biological plausibility metrics

## License

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

Attribution notice: If you use this software (code, models, or results) in your work, please provide appropriate credit to the authors of this repository.


