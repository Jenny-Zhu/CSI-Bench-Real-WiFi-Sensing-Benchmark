# Few-Shot Learning Pipeline for WiFi Sensing

This document provides comprehensive guidance on using the few-shot learning pipeline for adapting pre-trained WiFi sensing models to new environments, users, or devices with minimal examples.

## Overview

The few-shot learning pipeline enables adaptation of pre-trained models to new settings using only a few labeled examples from the target domain. The pipeline:

1. Loads the best pre-trained model for a given task
2. Performs few-shot adaptation to new settings (cross-env, cross-user, cross-device) 
3. Tunes the model using k-shot support data from the corresponding support splits
4. Evaluates the adapted model on the test splits
5. Saves the adapted model and performance metrics

## Requirements

- A pre-trained model in the `results/` directory for the task of interest
- Support data in the format of `support_cross_env.json`, `support_cross_user.json`, or `support_cross_device.json`
- Test data in the format of `test_cross_env.json`, `test_cross_user.json`, or `test_cross_device.json`

## Usage

### Local Execution

#### Using local_runner.py

```bash
python scripts/local_runner.py --pipeline fewshot --task MotionSourceRecognition --model vit
```

This will:
- Find the best pre-trained ViT model for MotionSourceRecognition
- Adapt it to all three cross-domain scenarios (environment, user, device)
- Save the results in the `results/fewshot/` directory

#### Advanced Options

```bash
python scripts/local_runner.py --pipeline fewshot \
                              --task MotionSourceRecognition \
                              --model vit \
                              --k_shots 5 \
                              --adaptation_lr 0.01 \
                              --adaptation_steps 10 \
                              --finetune_all
```

#### Using Configuration File

You can specify all parameters in the `local_default_config.json` file:

```json
{
  "pipeline": "fewshot",
  "task": "MotionSourceRecognition",
  "model": "vit",
  "fewshot": {
    "k_shots": 5,
    "adaptation_lr": 0.01,
    "adaptation_steps": 10,
    "finetune_all": false
  }
}
```

Then run:

```bash
python scripts/local_runner.py --config_file configs/local_default_config.json
```

#### Direct Execution

You can also run the few-shot pipeline script directly:

```bash
python scripts/train_fewshot_pipeline.py --task MotionSourceRecognition \
                                         --model vit \
                                         --k_shots 5 \
                                         --adaptation_lr 0.01 \
                                         --adaptation_steps 10
```

### SageMaker Execution

#### Using sagemaker_runner.py

```python
import sagemaker_runner

# Initialize the runner
runner = sagemaker_runner.SageMakerRunner()

# Run few-shot adaptation for a single task and model
runner.run_fewshot(
    task="MotionSourceRecognition",
    model_type="vit",
    k_shots=5,
    adaptation_lr=0.01,
    adaptation_steps=10
)
```

#### Batch Execution

```python
# Run few-shot adaptation for multiple tasks and models
runner.run_batch_fewshot(
    tasks=["MotionSourceRecognition", "HumanMotion"],
    models=["vit", "transformer"],
    k_shots=5,
    adaptation_lr=0.01,
    adaptation_steps=10
)
```

#### Command Line Interface

```bash
python sagemaker_runner.py fewshot --task MotionSourceRecognition --model vit

# For batch execution
python sagemaker_runner.py batch-fewshot --tasks MotionSourceRecognition HumanMotion --models vit transformer
```

## Configuration Parameters

The few-shot pipeline supports the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | Task name (e.g., MotionSourceRecognition) | - |
| `model` | Model type (e.g., vit, transformer) | - |
| `k_shots` | Number of examples per class for adaptation | 5 |
| `adaptation_lr` | Learning rate for adaptation | 0.01 |
| `adaptation_steps` | Number of gradient steps for adaptation | 10 |
| `finetune_all` | Whether to fine-tune all parameters or just the classifier | false |
| `batch_size` | Batch size for data loading | 32 |
| `win_len` | Window length for input data | 500 |
| `feature_size` | Feature size for input data | 232 |

## Output Structure

The pipeline saves results in the following structure:

```
results/
└── fewshot/
    └── {task}/
        └── {model}/
            └── fewshot_{scenario}_{run_id}/
                ├── {model}_{task}_fewshot_{scenario}.pth  # Adapted model
                ├── fewshot_{scenario}_results.json        # Performance metrics
                ├── {model}_{task}_fewshot_{scenario}_config.json  # Configuration
                └── confusion_matrix_{scenario}.png        # Confusion matrix visualization
```

Additionally, a summary file `{model}_{task}_fewshot_summary.json` is created with results from all scenarios.

## Examples

### Adapt ViT Model to New Environments

```bash
python scripts/local_runner.py --pipeline fewshot --task MotionSourceRecognition --model vit
```

### Compare Different Shot Values

```bash
python scripts/train_fewshot_pipeline.py --task MotionSourceRecognition \
                                         --model vit \
                                         --k_shots 1 \
                                         --adaptation_lr 0.01 \
                                         --adaptation_steps 10
```

Repeat with different `k_shots` values (1, 3, 5, 10) to compare performance.

### Fine-tune All Parameters

```bash
python scripts/local_runner.py --pipeline fewshot --task MotionSourceRecognition --model vit --finetune_all
```

### Using SageMaker for Large-Scale Experimentation

```python
import sagemaker_runner

runner = sagemaker_runner.SageMakerRunner()

# Run batch jobs for all tasks and models
runner.run_batch_fewshot(
    tasks=["MotionSourceRecognition", "HumanMotion", "DetectionandClassification"],
    models=["vit", "transformer", "resnet18"],
    k_shots=5,
    adaptation_lr=0.01,
    adaptation_steps=10
)
```

## Troubleshooting

1. **No pre-trained model found**: Ensure you have trained models in the `results/` directory for the specified task and model type.

2. **Missing support or test data**: Verify that your dataset includes the required support and test splits.

3. **Out of memory errors**: Try reducing the batch size or using a model with fewer parameters.

4. **Poor adaptation performance**: Experiment with different values for `adaptation_lr`, `adaptation_steps`, and `k_shots`.

## References

For more details on the underlying few-shot learning methods, refer to:

- `model/fewshot/models.py`: Implementation of the few-shot adaptation models
- `model/fewshot/trainer.py`: Implementation of the few-shot training process
- `engine/few_shot/adapter.py`: Implementation of the few-shot adaptation mechanism 