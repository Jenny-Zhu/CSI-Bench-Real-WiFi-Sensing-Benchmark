# WiFi Sensing Benchmark

A comprehensive benchmark and training system for WiFi sensing using CSI data.

## Overview

This repository provides a unified framework for training and evaluating deep learning models on WiFi Channel State Information (CSI) data for various sensing tasks. The framework supports both local execution and cloud-based training on AWS SageMaker.

## Installation and Setup

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended, but not required)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/WiAL-Real-WiFi-Sensing-Benchmark.git
   cd WiAL-Real-WiFi-Sensing-Benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Data organization:
   Organize your data in the following structure:
   ```
   data_directory/
   ├── tasks/
   │   ├── TaskName1/
   │   │   ├── train/
   │   │   │   └── data files (.h5)
   │   │   ├── val/
   │   │   │   └── data files (.h5)
   │   │   └── test/
   │   │       └── data files (.h5)
   │   └── ...
   └── ...
   ```

## Local Execution

The main entry point for local execution is `scripts/local_runner.py`. This script handles configuration loading, model training, and result storage.

### Configuration

Edit the local configuration file at `configs/local_default_config.json` to set your data path and other parameters:

```json
{
  "pipeline": "supervised",
  "training_dir": "/path/to/your/data/",
  "output_dir": "./results",
  "model": "mlp",
  "task": "YourTask",
  "win_len": 500,
  "feature_size": 232,
  "batch_size": 32,
  "epochs": 100,
  "test_splits": "all"
}
```

Key parameters:
- `pipeline`: Training pipeline type (`supervised` or `multitask`)
- `training_dir`: Path to your data directory
- `output_dir`: Directory to save results (default: `./results`)
- `model`: Model type to train (see Available Models)
- `task`: Task name (see Available Tasks)
- `batch_size`, `epochs`: Training parameters

### Running Models

Basic usage:
```bash
python scripts/local_runner.py
```

With custom configuration:
```bash
python scripts/local_runner.py --config_file configs/your_custom_config.json
```

### Available Models

- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer
- `patchtst`: PatchTST (Patch Time Series Transformer)
- `timesformer1d`: TimesFormer for 1D signals

### Available Tasks

- `MotionSourceRecognition`
- `HumanMotion`
- `HumanNonhuman`
- `NTUHumanID`
- `NTUHAR`
- `HumanID`
- `Widar`
- `ThreeClass`
- `DetectionandClassification`
- `Detection`

### Multi-Model Training

To train multiple models on a task:
```bash
python scripts/train_multi_model.py --all_models "mlp lstm transformer vit" --task_name MotionSourceRecognition
```

After completion, generate a summary of results:
```bash
python scripts/generate_summary_table.py --results_dir ./results
```

### Hyperparameter Tuning

The framework supports different hyperparameter tuning methods:

```bash
# Bayesian optimization with Optuna
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer --search_method optuna --num_trials 20

# Grid search
python scripts/hyperparameter_tuning.py --task_name HumanMotion --model_name lstm --search_method grid

# Random search
python scripts/hyperparameter_tuning.py --task_name HumanID --model_name vit --search_method random --num_trials 15
```

## Results Organization

Training results are saved with the following structure:

```
results/
├── task_name/                 # Name of the task
│   ├── model_name/            # Name of the model
│   │   ├── best_performance.json     # Record of best performance
│   │   ├── params_hash/              # Experiment identifier
│   │   │   ├── model_task_config.json           # Model configuration
│   │   │   ├── model_task_results.json          # Training metrics
│   │   │   ├── model_task_summary.json          # Performance summary
│   │   │   ├── model_task_test_confusion.png    # Confusion matrix
│   │   │   ├── classification_report_test.csv   # Classification metrics
│   │   │   └── checkpoint/                      # Saved model weights
│   │   └── hyperparameter_tuning/    # Results from hyperparameter tuning
│   ├── performance_summary.csv       # Summary table for this task
│   └── all_models_summary.json       # Combined results of all models
└── ...
```

## Advanced Features

### Multi-Task Learning

Train a model on multiple tasks with shared backbone:
```bash
python scripts/local_runner.py --config_file configs/multitask_config.json
```

### Few-Shot Learning

Enable few-shot adaptation for domain generalization:
```bash
python scripts/local_runner.py --model transformer --task MotionSourceRecognition --enable_few_shot --k_shot 5 --inner_lr 0.01 --num_inner_steps 10
```

## SageMaker Integration (Optional)

The repository also supports training on AWS SageMaker. See the [SageMaker Guide](docs/sagemaker_guide.md) for details.

Basic SageMaker usage:
```python
from scripts import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner(config)
runner.run_batch_by_task(
    tasks=['MotionSourceRecognition', 'HumanID'], 
    models=['vit', 'transformer', 'resnet18']
)
```

## Citation

If you use this code in your research, please cite:
```
@article{wifi_sensing_benchmark,
  title={WiFi Sensing Benchmark: A Comprehensive Evaluation Framework for WiFi Sensing},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
