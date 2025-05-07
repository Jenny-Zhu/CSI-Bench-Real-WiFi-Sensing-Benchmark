# Few-Shot Learning for WiFi Sensing

This module implements few-shot learning for WiFi sensing, allowing models to quickly adapt to new environments, users, or devices using only a small number of labeled examples.

## Overview

Few-shot learning addresses the problem of poor generalization in WiFi sensing models when they are exposed to new settings that differ from the training environment. Instead of retraining a model from scratch with a large dataset, few-shot learning enables adaptation using just a handful of examples from the new environment.

## How it Works

1. **Pre-trained Model**: We start with a model that has been trained on a large dataset in a supervised manner.
   
2. **Support Set**: We collect a small labeled dataset (support set) from the new environment, typically just a few examples per class (e.g., 1-10 examples).
   
3. **Adaptation**: The model is fine-tuned on the support set for a small number of steps to adapt to the new environment. By default, we only update the classification head to prevent overfitting, but the entire model can be fine-tuned if specified.
   
4. **Evaluation**: The adapted model is then evaluated on a test set from the new environment.

## Components

- **FewShotAdaptiveModel**: A model that wraps a pre-trained model and provides functionality for few-shot adaptation.
  
- **FewShotTrainer**: A trainer class that handles the adaptation process and evaluation.
  
- **train_fewshot.py**: A script for training and evaluating few-shot models.

## Usage

### Command Line

```bash
python scripts/train_fewshot.py --model_path /path/to/pretrained/model.pt --model vit --task_name MotionSourceRecognition --support_split val_id --query_split test_cross_env
```

Or using the local runner:

```bash
python scripts/local_runner.py --pipeline fewshot --model_path /path/to/pretrained/model.pt --model vit --task MotionSourceRecognition
```

### Important Parameters

- `--model_path`: Path to the pre-trained model checkpoint (required)
- `--adaptation_lr`: Learning rate for few-shot adaptation (default: 0.01)
- `--adaptation_steps`: Number of adaptation steps (default: 10)
- `--k_shots`: Number of examples per class for adaptation (default: 5)
- `--finetune_all`: If set, fine-tune all parameters instead of just the classifier
- `--support_split`: Split to use for the support set (default: val_id)
- `--query_split`: Split to use for evaluation (default: test_cross_env)
- `--eval_shots`: If set, evaluate different k-shot values (1, 3, 5, 10)

## Example

The following example shows how to use few-shot learning to adapt a pre-trained ViT model to a new environment:

```python
from model.fewshot import FewShotAdaptiveModel, FewShotTrainer
import torch

# Load pre-trained model
fewshot_model = FewShotAdaptiveModel.from_pretrained(
    model_path='results/MotionSourceRecognition/vit/best_model.pt',
    model_type='vit',
    adaptation_lr=0.01,
    adaptation_steps=10
)

# Create trainer
trainer = FewShotTrainer(
    base_model=fewshot_model,
    support_loader=support_loader,  # Few examples from new environment
    query_loader=query_loader,      # Test set from new environment
    adaptation_steps=10,
    adaptation_lr=0.01,
    save_path='results/fewshot'
)

# Compare performance with and without adaptation
results = trainer.compare_with_without_adaptation()

# Evaluate different support set sizes
k_shot_results = trainer.evaluate_support_set_sizes(k_shots_list=[1, 3, 5, 10])
```

## Results Interpretation

The results include:

- **Accuracy comparison** with and without few-shot adaptation
- **F1-score comparison** with and without few-shot adaptation
- **Confusion matrices** for both approaches
- **K-shot performance curves** showing how performance changes with different numbers of examples

Look for the following output files:
- `fewshot_comparison_summary.json`: Summary of comparison results
- `fewshot_confusion_matrices.png`: Confusion matrices visualization
- `fewshot_kshot_results.json`: Results with different k-shot values
- `fewshot_kshot_performance.png`: Performance curves with varying k values 