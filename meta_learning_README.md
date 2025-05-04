# Meta-Learning for WiFi Sensing

This implementation provides a meta-learning framework for WiFi sensing tasks, using the WiAL-Real-WiFi-Sensing-Benchmark. It supports a variety of model architectures and evaluation scenarios.

## Key Features

- **Multiple Model Architectures**: MLP, LSTM, ResNet18, Transformer, and Vision Transformer (ViT)
- **Model-Agnostic Meta-Learning (MAML)**: Fast adaptation to new domains
- **Cross-Domain Testing**: Evaluate generalization to unseen users, environments, and devices
- **Few-Shot Learning**: Test with limited examples (1-shot and 5-shot)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/WiAL-Real-WiFi-Sensing-Benchmark.git
cd WiAL-Real-WiFi-Sensing-Benchmark

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
├── meta_model.py                 # Model implementations (MLP, LSTM, ResNet18, Transformer, ViT)
├── meta_learning_data.py         # Data loading utilities
├── train_meta_all_models.py      # Training script
├── test_meta_models.py           # Testing script
├── compare_meta_models.py        # Comparison script
├── run_meta_experiments.py       # Experiment runner
└── wifi_benchmark_dataset/       # Dataset directory
    └── tasks/
        └── MotionSourceRecognition/
            └── meta_splits/      # Meta-learning task definitions
```

## Usage

### Training a Model

To train a model with meta-learning:

```bash
python train_meta_all_models.py --model_type vit --data_dir wifi_benchmark_dataset --task_name MotionSourceRecognition
```

Available model types:
- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer

### Testing a Model

To test a trained model on various scenarios:

```bash
python test_meta_models.py --checkpoint results/meta_learning/vit_MotionSourceRecognition/best_model.pth --test_type cross_user
```

Available test types:
- `test`: In-distribution testing
- `cross_env`: Cross-environment testing
- `cross_user`: Cross-user testing
- `cross_device`: Cross-device testing
- `adapt_1shot`: 1-shot adaptation
- `adapt_5shot`: 5-shot adaptation
- Plus combinations like `cross_user_adapt_1shot`

### Comparing Models

To compare multiple models across test scenarios:

```bash
python compare_meta_models.py --results_dir results/meta_testing
```

### Running Full Experiments

To run a full set of experiments across multiple models and test types:

```bash
python run_meta_experiments.py --num_iterations 2000 --models vit resnet18 transformer
```

## Meta-Learning Tasks

The `wifi_benchmark_dataset/tasks/MotionSourceRecognition/meta_splits/` directory contains JSON files defining meta-learning tasks:

- `train_tasks.json`: Tasks for meta-training
- `val_tasks.json`: Tasks for meta-validation
- `test_tasks.json`: In-distribution test tasks
- `test_cross_env_tasks.json`: Cross-environment test tasks
- `test_cross_user_tasks.json`: Cross-user test tasks
- `test_cross_device_tasks.json`: Cross-device test tasks
- Various adaptation task sets for 1-shot and 5-shot learning

Each task consists of a support set (for adaptation) and a query set (for evaluation).

## Example: Small Test Run

To do a quick test run with a small number of iterations:

```bash
# Train with fewer iterations
python train_meta_all_models.py --model_type mlp --num_iterations 100 --save_dir results/quick_test

# Test on a single test type
python test_meta_models.py --checkpoint results/quick_test/mlp_MotionSourceRecognition/best_model.pth --test_type test
```

## Customizing Models

You can customize the model architectures by modifying `meta_model.py`. The key parameters for each model are:

- `win_len`: Window length for CSI data (default: 232)
- `feature_size`: Feature size for CSI data (default: 500)
- `emb_dim`: Embedding dimension (default: 128)
- `num_classes`: Number of classes (default: same as n_way)
- `dropout`: Dropout rate (default: 0.1)

## Citation

If you use this code, please cite:

```
@article{wial2023,
  title={WiAL: Real-World WiFi Sensing Benchmark},
  author={...},
  journal={...},
  year={2023}
}
``` 