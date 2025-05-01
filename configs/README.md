# Configuration Files

This directory contains configuration files for the WiFi sensing benchmark.

## Model Configuration Files

Model-specific configuration files:

- `mlp_config.json`: Configuration for Multi-Layer Perceptron model
- `lstm_config.json`: Configuration for Long Short-Term Memory model
- `resnet18_config.json`: Configuration for ResNet-18 CNN model
- `transformer_config.json`: Configuration for Transformer-based model
- `vit_config.json`: Configuration for Vision Transformer model

## Default Configuration Files

System-wide default configuration files:

- `local_default_config.json`: Default configuration for local runs
- `sagemaker_default_config.json`: Default configuration for SageMaker runs

## Configuration Structure

### Common Parameters

Configuration files typically include the following parameters:

```json
{
  "model_name": "transformer",
  "task": "MotionSourceRecognition",
  "batch_size": 32,
  "num_epochs": 10,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "win_len": 232,
  "feature_size": 500,
  "results_subdir": "transformer_msr",
  "training_dir": "wifi_benchmark_dataset",
  "output_dir": "./results"
}
```

### Default Configuration Files

The default configuration files include additional system-wide settings:

```json
{
  "pipeline": "supervised",
  "training_dir": "wifi_benchmark_dataset",
  "test_dirs": [],
  "output_dir": "./results",
  "mode": "csi",
  "freeze_backbone": false,
  "integrated_loader": true,
  "task": "MotionSourceRecognition",
  "win_len": 250,
  "feature_size": 98,
  "seed": 42,
  "batch_size": 8,
  "num_epochs": 10,
  "model_name": "transformer",
  "task_class_mapping": {
    "HumanNonhuman": 2,
    "MotionSourceRecognition": 4,
    "NTUHumanID": 15,
    ...
  },
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
  "available_tasks": [
    "MotionSourceRecognition",
    "HumanMotion",
    ...
  ]
}
```

The SageMaker configuration additionally includes AWS-specific settings:

```json
{
  "s3_data_base": "s3://rnd-sagemaker/Data/Benchmark/",
  "s3_output_base": "s3://rnd-sagemaker/Benchmark_Log/",
  "instance_type": "ml.g4dn.xlarge",
  "instance_count": 1,
  "framework_version": "1.12.1",
  "py_version": "py38",
  "base_job_name": "wifi-sensing-supervised",
  "batch_wait_time": 30,
  "batch_mode": "by-task",
  "task_test_dirs": {
    "demo": ["test/"]
  }
}
```

## Usage

When running the benchmark, configurations are loaded in the following order:

1. Default configuration from `local_default_config.json` or `sagemaker_default_config.json`
2. Model-specific configuration (e.g., `transformer_config.json`)
3. Command-line arguments

This allows for flexibility in overriding settings at different levels.

## Examples

### Local Training

```bash
# Basic usage with defaults from local_default_config.json
python scripts/local_runner.py

# Specify model and task
python scripts/local_runner.py --model lstm --task HumanMotion

# Override with specific model configuration file
python scripts/local_runner.py --config_file configs/transformer_config.json

# Override specific parameters
python scripts/local_runner.py --model vit --task MotionSourceRecognition --epochs 20 --batch_size 64
```

### SageMaker Training

```python
# In a Jupyter notebook
import sagemaker_runner

# Basic usage with defaults from sagemaker_default_config.json
runner = sagemaker_runner.SageMakerRunner()
runner.run_supervised()

# Specify model and task
runner.run_supervised(model_name='lstm', task='HumanMotion')

# Batch run multiple models and tasks
runner.run_batch(tasks=['HumanMotion', 'MotionSourceRecognition'], models=['transformer', 'vit'])
```

## Modifying Configurations

To modify default configurations:

1. Edit the appropriate JSON file in the `configs/` directory
2. For one-time changes, use command-line arguments instead
3. For persistent custom configurations, create a new JSON file and specify it with `--config_file`

When you first run a specific model+task combination, a configuration file will be automatically saved in the `configs/` directory (e.g., `lstm_humanmotion_config.json`), which you can reuse or modify for future runs. 