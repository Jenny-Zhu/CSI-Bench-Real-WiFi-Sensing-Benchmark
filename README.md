# WiFi Sensing Benchmark

A comprehensive benchmark and training system for WiFi sensing using CSI data.

## Project Structure

```
├── configs/                  # Model configuration files
│   ├── local_default_config.json    # Default configuration for local training
│   ├── sagemaker_default_config.json # Default configuration for SageMaker
│   ├── multitask_config.json        # Configuration for multi-task learning
│   └── transformer_config.json      # Configuration for transformer model
├── scripts/                  # Training and utility scripts
│   ├── local_runner.py       # Main entry point for local training
│   ├── sagemaker_runner.py   # Runner for SageMaker training jobs
│   ├── entry_script.py       # SageMaker entry script
│   ├── train_multi_model.py  # Script for training multiple models
│   ├── train_supervised.py   # Implementation of training loop
│   ├── train_fewshot.py      # Few-shot learning training script
│   ├── train_multitask_adapter.py # Multi-task adapter training
│   ├── hyperparameter_tuning.py # Hyperparameter tuning script
│   └── generate_summary_table.py # Generate performance summary tables
├── model/                    # Model implementations
│   ├── supervised/           # Supervised learning models
│   ├── multitask/            # Multi-task learning models
│   └── fewshot/              # Few-shot learning models
├── engine/                   # Training engines
│   ├── supervised/           # Supervised learning trainers
│   └── fewshot/              # Few-shot learning trainers
├── load/                     # Data loading utilities
│   ├── supervised/           # Supervised learning data loaders
│   └── fewshot/              # Few-shot learning data loaders
├── data/                     # Data processing modules
│   ├── datasets/             # Dataset definitions
│   │   └── csi/              # Channel State Information datasets
│   └── preprocessing/        # Data preprocessing utilities
├── util/                     # Utility functions
│   └── converters/           # Data format conversion utilities
├── wifi_benchmark_dataset/   # Dataset directory
│   └── tasks/                # Different WiFi sensing tasks
└── results/                  # Training results and models
```

## Latest Features

The benchmark system has been streamlined and optimized with the following features:

1. **Improved Storage Architecture**:
   - Uses a `results/task/model/experiment_id/` structure, where `experiment_id` is a unique identifier based on parameter hash
   - Experiments with the same parameters will overwrite instead of creating new folders
   - Each model directory contains a `best_performance.json` file to record the best performance across all experiments

2. **Automatic Hyperparameter Tuning**:
   - Supports three search methods:
     - Grid search: Systematically search all parameter combinations
     - Random search: Randomly sample parameter combinations
     - Bayesian optimization: Implemented with the Optuna library

3. **Flexible Training and Evaluation Workflow**:
   - Separated storage of results and generation of summary tables
   - Training scripts generate summary JSON files, which can be used later to generate summary tables
   - Run multiple experiments or hyperparameter tuning first, then generate summary tables of all results

4. **Enhanced Result Summarization**:
   - `generate_summary_table.py` reads best performance from each model's `best_performance.json` file
   - Automatically uses the best performance of each model for summaries

5. **SageMaker Integration**:
   - Enhanced `sagemaker_runner.py` for batch task processing, with each task using a single instance
   - Improved parameter handling and S3 path management
   - Increased default EBS volume size for sufficient space
   - All scripts use English for comments and logging

## Configuration

The benchmark system uses JSON configuration files in the `configs/` directory:

- `local_default_config.json`: Default configuration for local training
- `sagemaker_default_config.json`: Default configuration for SageMaker
- `multitask_config.json`: Configuration for multi-task learning
- `transformer_config.json`: Model-specific configuration

### Custom Configuration

You can create custom configuration files for specific model-task combinations. When you run a model for the first time, a configuration file will be automatically created in the `configs/` directory.

Example of a configuration file:

```json
{
  "model": "transformer",
  "task_name": "MotionSourceRecognition",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "win_len": 500,
  "feature_size": 232,
  "d_model": 128,
  "dropout": 0.1,
  "patience": 15,
  "test_splits": "test"
}
```

## Running Models Locally

You can easily train different models using the local runner script:

```bash
python scripts/local_runner.py --model [model_name] --task [task_name]
```

### Available Models

- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer

### Available Tasks

- `MotionSourceRecognition`
- `HumanMotion`
- `DetectionandClassification`
- `HumanID`
- `NTUHAR`
- `HumanNonhuman`
- `NTUHumanID`
- `Widar`
- `ThreeClass`
- `Detection`

### Examples

```bash
# Train an LSTM model for MotionSourceRecognition
python scripts/local_runner.py --model lstm --task MotionSourceRecognition

# Train a transformer model with custom parameters
python scripts/local_runner.py --model transformer --task HumanID --epochs 20 --batch_size 64
```

### Using Configuration Files

You can also specify a configuration file:

```bash
python scripts/local_runner.py --config_file configs/custom_config.json
```

### Command-Line Arguments

You can override default configurations with command-line arguments:

```bash
python scripts/local_runner.py --model lstm --task MotionSourceRecognition --epochs 20 --batch_size 64 --output_dir ./custom_results
```

## Multi-Model Training

You can use the `train_multi_model.py` script to train multiple models at once:

```bash
# Train multiple models on MotionSourceRecognition task
python scripts/train_multi_model.py --all_models "mlp lstm transformer vit" --task_name MotionSourceRecognition
```

After completion, run the summary table generation script:

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

## Hyperparameter Tuning

The system provides powerful hyperparameter tuning functionality:

```bash
# Hyperparameter tuning using Optuna Bayesian optimization
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer --search_method optuna --num_trials 20

# Hyperparameter tuning using grid search
python scripts/hyperparameter_tuning.py --task_name HumanMotion --model_name lstm --search_method grid

# Hyperparameter tuning using random search
python scripts/hyperparameter_tuning.py --task_name HumanID --model_name vit --search_method random --num_trials 15
```

### Custom Parameter Ranges

You can define custom parameter search ranges using command line arguments:

```bash
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer \
    --search_method random --num_trials 20 \
    --learning_rates "0.001,0.0005,0.0001" \
    --batch_sizes "16,32,64" \
    --dropout_rates "0.1,0.3,0.5"
```

For random search and Bayesian optimization, you can also specify continuous parameter ranges:

```bash
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer \
    --search_method optuna --num_trials 20 \
    --lr_min 0.0001 --lr_max 0.01 \
    --batch_size_min 16 --batch_size_max 128 \
    --dropout_min 0.0 --dropout_max 0.5
```

## SageMaker Integration

For large-scale training on AWS SageMaker, we provide a specialized runner that allows you to train models in the cloud.

### Running on SageMaker

You can use the SageMaker runner in a Python script or Jupyter notebook:

```python
from scripts import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner(load_config())
runner.run_batch_by_task(
    tasks=['MotionSourceRecognition', 'HumanID'], 
    models=['vit', 'transformer', 'resnet18']
)
```

Or from the command line:

```bash
python scripts/sagemaker_runner.py --task MotionSourceRecognition,HumanID
```

### SageMaker Configuration

The SageMaker configuration can be customized by editing the `configs/sagemaker_default_config.json` file. This includes:

- S3 paths for data and output
- Default instance types
- Available tasks and models
- Training parameters

Example SageMaker configuration:

```json
{
  "s3_data_base": "s3://my-bucket/Data/Benchmark/",
  "s3_output_base": "s3://my-bucket/Benchmark_Log/",
  "instance_type": "ml.g4dn.xlarge",
  "instance_count": 1,
  "volume_size": 30,
  "batch_wait_time": 30,
  "available_tasks": [
    "MotionSourceRecognition",
    "HumanMotion",
    "HumanID"
  ],
  "available_models": [
    "mlp",
    "lstm",
    "transformer",
    "vit",
    "resnet18"
  ],
  "base_job_name": "wifi-sensing",
  "adaptive_path": true,
  "try_all_paths": true
}
```

### Data Organization for SageMaker

When using SageMaker, your data should be organized in S3 as follows:

```
s3://my-bucket/Data/Benchmark/
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

The `sagemaker_runner.py` script will handle uploading the code to SageMaker and configuring the training environment.

## Results Structure

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
├── full_summary.csv                  # Comprehensive summary across all tasks
├── accuracy_summary.csv              # Summary focused on accuracy metrics
└── f1_score_summary.csv              # Summary focused on F1 score metrics
```

## Requirements

- Python 3.7+
- PyTorch 1.12.1+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, scipy
- h5py, mat73
- einops
- transformers, accelerate (for adapter models)
- Optional: Optuna (for Bayesian optimization)
