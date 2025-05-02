# WiFi Sensing Benchmark

A comprehensive benchmark and training system for WiFi sensing using CSI data.

## Project Structure

```
├── configs/                  # Model configuration files
│   ├── lstm_config.json
│   ├── mlp_config.json
│   ├── resnet18_config.json
│   ├── transformer_config.json
│   └── vit_config.json
├── scripts/                  # Training and utility scripts
│   ├── local_runner.py       # Main entry point for training
│   └── train_supervised.py   # Implementation of training loop
├── model/                    # Model implementations
│   └── supervised/           # Supervised learning models
├── engine/                   # Training engines
│   └── supervised/           # Supervised learning trainers
├── load/                     # Data loading utilities
│   └── supervised/           # Supervised learning data loaders
├── wifi_benchmark_dataset/   # Dataset directory
│   └── tasks/                # Different WiFi sensing tasks
└── results/                  # Training results and models
```

## Training Models

You can easily train different models using our main entry point script:

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
python scripts/local_runner.py --config_file configs/transformer_humanmotion_config.json
```

The first time you run a model+task combination, a configuration file will be automatically created in the `configs/` directory, which you can reuse or modify for future runs.

## Pipeline Options

The benchmark supports two training pipelines:

1. **Supervised Learning**: The default pipeline for training models
2. **Meta-Learning**: A more advanced pipeline for few-shot learning (under development)

To specify a pipeline:

```bash
python scripts/local_runner.py --pipeline supervised --model vit --task HumanMotion
python scripts/local_runner.py --pipeline meta  # Meta-learning pipeline is under development
```

## Configuration

You can override default configurations with command-line arguments:

```bash
python scripts/local_runner.py --model lstm --task MotionSourceRecognition --epochs 20 --batch_size 64 --output_dir ./custom_results
```

## Results

Training results are saved in the `results/` directory with the following structure:

```
results/
├── task_name/                 # Name of the task (e.g., MotionSourceRecognition)
│   ├── model_name/            # Name of the model (e.g., transformer)
│   │   ├── model_task_config.json           # Model configuration
│   │   ├── model_task_results.json          # Training metrics and evaluation results
│   │   ├── model_task_summary.json          # Performance summary with accuracy and F1 scores
│   │   ├── model_task_test_confusion.png    # Confusion matrix for test data
│   │   ├── classification_report_test.csv   # Detailed classification metrics for test data
│   │   └── checkpoint/                      # Saved model weights
│   ├── performance_summary.csv              # Summary table of all models for this task
│   └── all_models_summary.json              # Combined results of all models for this task
├── full_summary.csv                         # Comprehensive summary of all models across all tasks
├── accuracy_summary.csv                     # Summary table focused on accuracy metrics
└── f1_score_summary.csv                     # Summary table focused on F1 score metrics
```

Each model's results are stored in a dedicated directory with the structure: `results/task_name/model_name/`. This organization makes it easier to compare different models for the same task or track a specific model's performance across different tasks.

The files include:
- Trained model weights
- Training metrics and logs
- Confusion matrices
- Classification reports
- Summary tables with overall accuracy and F1 scores

### Generating Summary Tables

To generate summary tables that aggregate results across all models and tasks:

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

This will create several CSV tables in the results directory:
- `full_summary.csv`: Complete metrics for all models and tasks
- `accuracy_summary.csv`: Focused on accuracy metrics
- `f1_score_summary.csv`: Focused on F1 score metrics
- `compact_summary.csv`: Key metrics in a condensed format
- `pivot_test_accuracy.csv`: Comparison of test accuracy across models and tasks
- `pivot_test_f1_score.csv`: Comparison of test F1 scores across models and tasks

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib, Seaborn
