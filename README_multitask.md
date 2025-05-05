# Multitask Learning for WiFi Sensing Benchmark

This document describes how to use the multitask learning pipeline in the WiFi Sensing Benchmark framework.

## Overview

Multitask learning allows a single model to learn multiple tasks simultaneously by sharing a common backbone model and adding task-specific adapters. This approach can lead to improved performance compared to training separate models for each task, especially when the tasks are related.

The implementation uses a Transformer backbone model with:
- Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- Task-specific adapters to specialize the model for each task
- Task-specific classification heads

## Requirements

- PyTorch 1.9+
- pandas, numpy, matplotlib
- The standard WiFi Sensing Benchmark environment

## Running Locally

You can run the multitask learning pipeline locally using the `local_runner.py` script:

```bash
# Train on multiple tasks with transformer model
python scripts/local_runner.py --pipeline multitask --tasks "MotionSourceRecognition,HumanMotion" --model transformer

# Specify more options
python scripts/local_runner.py --pipeline multitask \
  --tasks "MotionSourceRecognition,HumanMotion,HumanID" \
  --model transformer \
  --batch_size 16 \
  --epochs 20 \
  --output_dir ./results
```

## Running on SageMaker

For distributed training on AWS SageMaker, use the `sagemaker_runner.py` script:

```python
import sagemaker_runner

# Create runner
runner = sagemaker_runner.SageMakerRunner()

# Run simple multitask job
runner.run_multitask(
    tasks=["MotionSourceRecognition", "HumanMotion"],
    model_type="transformer"
)

# Run batch of multitask jobs with different task combinations
task_groups = [
    ["MotionSourceRecognition", "HumanMotion"],
    ["HumanNonhuman", "HumanID"],
    ["NTUHAR", "NTUHumanID"]
]

runner.run_batch_multitask(
    task_groups=task_groups,
    model_type="transformer",
    instance_type="ml.g4dn.xlarge"
)
```

You can also use the command line interface:

```bash
# Run multitask learning on SageMaker
python sagemaker_runner.py --pipeline multitask \
  --task-groups "MotionSourceRecognition,HumanMotion" "HumanNonhuman,HumanID" \
  --model-type transformer \
  --instance-type ml.g4dn.xlarge
```

## Configuration

The multitask learning pipeline can be configured using the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| tasks | Comma-separated list of tasks to train on | "MotionSourceRecognition" |
| model_type | Base model type | "transformer" |
| data_dir | Directory containing training data | "wifi_benchmark_dataset" |
| batch_size | Batch size for training | 8 |
| epochs | Number of epochs to train | 10 |
| lr | Learning rate | 1e-4 |
| win_len | Window length | 250 |
| feature_size | Feature size | 98 |
| lora_r | LoRA rank | 8 |
| lora_alpha | LoRA alpha | 32 |
| lora_dropout | LoRA dropout | 0.05 |
| patience | Patience for early stopping | 10 |
| save_dir | Directory to save results | "./results/multitask" |

## Extending the Pipeline

### Adding New Tasks

To add a new task to the multitask pipeline:

1. Ensure the task is defined in the dataset loaders
2. Add the task to the `task_class_mapping` in the configuration file
3. Pass the task name in the `tasks` parameter when running the pipeline

### Supporting Other Model Architectures

Currently, the multitask pipeline only supports the Transformer model architecture. To add support for other architectures:

1. Modify the `MultiTaskAdapterModel` in `model/multitask/models.py` to handle different backbone models
2. Update the extraction of embeddings in `train_multitask_adapter.py` for each model type
3. Test the new architecture with a simple task combination

## Results and Evaluation

The multitask learning pipeline saves the following results:

- Training and validation metrics for each task
- Confusion matrices for each task
- Saved adapter weights for the final model

Results are saved in the specified `save_dir` directory, with a subdirectory for each experiment.

## Comparison with Supervised Learning

To compare multitask learning with supervised learning:

1. Train models using both pipelines on the same tasks
2. Compare the performance metrics on the test sets
3. Analyze the results to determine if multitask learning improves performance 