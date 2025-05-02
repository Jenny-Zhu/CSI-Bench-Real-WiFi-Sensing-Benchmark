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
│   ├── train_supervised.py   # Implementation of training loop
│   ├── hyperparameter_tuning.py # Hyperparameter tuning script
│   └── generate_summary_table.py # Generate performance summary tables
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

## Latest Features

We have made the following improvements to the system:

1. **Improved Storage Architecture**:
   - Now uses a `results/task/model/experiment_id/` structure, where `experiment_id` is a unique identifier generated based on parameter hash
   - Experiments with the same parameters will overwrite instead of creating new folders, making it easy to conduct multiple attempts
   - Added a `best_performance.json` file in each model directory to record the best performance of the model across all experiments

2. **Automatic Hyperparameter Tuning**:
   - Added `scripts/hyperparameter_tuning.py` script with support for three search methods:
     - Grid search: Systematically search all parameter combinations
     - Random search: Randomly sample parameter combinations within a given range
     - Bayesian optimization: Implemented with the Optuna library

3. **Flexible Training and Evaluation Workflow**:
   - Separated the storage of results and generation of summary tables, making training more flexible
   - Modified `train_multi_model.py` to only generate summary JSON files, no longer automatically generating summary tables
   - The modified workflow allows you to run multiple experiments or hyperparameter tuning first, then generate summary tables of all results at once

4. **Enhanced Result Summarization**:
   - Updated `generate_summary_table.py` to now read best performance from each model's `best_performance.json` file
   - No longer need to manually select which experiments to include in the summary, the system automatically uses the best performance of each model

5. **SageMaker Integration Improvements**:
   - Enhanced `sagemaker_runner.py` to focus on batch task processing, with each task using a single instance to run multiple models
   - Fixed job name length issue (SageMaker has a 63-character limit)
   - Improved S3 path handling for better data access in SageMaker environment
   - Added better parameter support for training script
   - Increased default EBS volume size to 30GB to ensure sufficient space
   - All scripts use English for comments and logging for better international readability

These modifications make the system more efficient to run on AWS SageMaker, improving resource utilization and cost-effectiveness by running all models for each task on a single instance.

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
│   │   ├── best_performance.json     # Record of best performance across all experiments
│   │   ├── params_hash/              # Experiment identifier based on parameter hash
│   │   │   ├── model_task_config.json           # Model configuration
│   │   │   ├── model_task_results.json          # Training metrics and evaluation results
│   │   │   ├── model_task_summary.json          # Performance summary with accuracy and F1 scores
│   │   │   ├── model_task_test_confusion.png    # Confusion matrix for test data
│   │   │   ├── classification_report_test.csv   # Detailed classification metrics for test data
│   │   │   └── checkpoint/                      # Saved model weights
│   │   ├── params_hash2/              # Another experiment with different parameters
│   │   └── hyperparameter_tuning/    # Results from hyperparameter tuning
│   ├── performance_summary.csv       # Summary table of all models for this task
│   └── all_models_summary.json       # Combined results of all models for this task
├── full_summary.csv                  # Comprehensive summary of all models across all tasks
├── accuracy_summary.csv              # Summary table focused on accuracy metrics
└── f1_score_summary.csv              # Summary table focused on F1 score metrics
```

Each experiment is stored in a directory named with a parameter hash. The `best_performance.json` file in each model directory tracks the best experiment results with links to the corresponding experiment directory.

## Hyperparameter Tuning

The system provides powerful hyperparameter tuning functionality to help you find the best model parameter configurations:

```bash
# Hyperparameter tuning using Optuna Bayesian optimization
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer --search_method optuna --num_trials 20

# Hyperparameter tuning using grid search
python scripts/hyperparameter_tuning.py --task_name HumanMotion --model_name lstm --search_method grid

# Hyperparameter tuning using random search
python scripts/hyperparameter_tuning.py --task_name HumanID --model_name vit --search_method random --num_trials 15
```

### Tuning Options

- **Search Methods**:
  - `grid`: Grid search, exhaustively searching all parameter combinations (suitable for smaller search spaces)
  - `random`: Random search, randomly sampling from the parameter space (suitable for larger search spaces)
  - `optuna`: Bayesian optimization implemented using the Optuna library (most efficient, requires `pip install optuna`)

- **Custom Parameter Ranges**:
  
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

Tuning results are saved in the `hyperparameter_tuning/` directory under each model and task combination, including a detailed CSV file of all trial results and a JSON summary of the best parameters found.

### Generating Summary Tables

To generate tables summarizing the results of different models and tasks, run:

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

This will create several CSV tables in the results directory:
- `full_summary.csv`: Complete metrics for all models and tasks
- `accuracy_summary.csv`: Focused on accuracy metrics
- `f1_score_summary.csv`: Focused on F1 score metrics
- `compact_summary.csv`: Key metrics in a more compact format
- `test_accuracy_pivot.csv`: Comparison of test accuracy across different models and tasks
- `test_f1_score_pivot.csv`: Comparison of test F1 scores across different models and tasks

### Multi-Model Training

You can use the `train_multi_model.py` script to train multiple models at once:

```bash
# Train multiple models on MotionSourceRecognition task
python train_multi_model.py --all_models "mlp lstm transformer vit" --task_name MotionSourceRecognition
```

After completion, you will need to manually run the summary table generation script:

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Optuna (for Bayesian optimization hyperparameter tuning)

## SageMaker Integration

For large-scale training on AWS SageMaker, we provide a specialized runner that allows you to train models in the cloud. The SageMaker runner has been optimized to run multiple models on a single instance for each task, improving resource utilization and reducing costs.

### Key Features

- Run multiple models for each task on a single SageMaker instance
- Automated task-based batch processing
- Customizable instance types and training parameters
- Comprehensive job tracking and result summarization
- Optimized S3 path handling for data access
- Automatic fix for SageMaker's 63-character job name limit

### Usage

You can use the SageMaker runner in a Python script or Jupyter notebook:

```python
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner()
runner.run_batch_by_task(
    tasks=['MotionSourceRecognition', 'HumanID'], 
    models=['vit', 'transformer', 'resnet18']
)
```

Or from the command line:

```bash
python sagemaker_runner.py --tasks MotionSourceRecognition HumanID --models vit transformer resnet18
```

### Requirements

- AWS account with SageMaker access
- S3 bucket named "rnd-sagemaker" (configurable)
- IAM role with appropriate permissions

### Additional Options

- `--mode`: Data modality ('csi' or 'acf')
- `--instance-type`: SageMaker instance type (default: ml.g4dn.xlarge)
- `--batch-wait`: Wait time between batch job submissions in seconds
- `--volume-size`: Size of the EBS volume in GB (default: 30)

### SageMaker Configuration

The SageMaker configuration can be customized by editing the `configs/sagemaker_default_config.json` file. This includes:

- S3 paths for data and output
- Default instance types
- Available tasks and models
- Training parameters

### Data Organization for SageMaker

When using SageMaker, your data should be organized in S3 as follows:

```
s3://rnd-sagemaker/Data/Benchmark/
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

### Output Structure

The SageMaker runner maintains the same output structure as the local training pipeline:

```
s3://rnd-sagemaker/Benchmark_Log/
  ├── TaskName1/
  │   ├── ModelName1/
  │   │   ├── experiment_id_1/
  │   │   │   ├── model.pth
  │   │   │   ├── model_summary.json
  │   │   │   └── training_results.csv
  │   │   ├── ...
  │   │   └── best_performance.json
  │   └── ...
  └── ...
```

### Monitoring SageMaker Jobs

You can monitor your SageMaker jobs in several ways:

1. **SageMaker Console**: 
   - View all training jobs in the AWS SageMaker console
   - Monitor metrics in real-time
   - Check logs for errors or progress

2. **CloudWatch Logs**:
   - Detailed logs are available in CloudWatch
   - Search for specific error messages
   - Set up alerts for job completion or failures

3. **Local Job Summaries**:
   - The `batch_summaries/` directory contains summaries of all submitted jobs
   - Each batch has both text and JSON format summaries
   - Reference these files to track job names and S3 output locations

### Debugging SageMaker Jobs

If you encounter issues with SageMaker training:

1. Check CloudWatch logs for error messages
2. Verify your S3 data paths and permissions
3. Confirm that your IAM role has sufficient permissions
4. Try running with a smaller dataset locally first to validate your pipeline

The improved debugging information in `train_multi_model.py` will help identify data loading issues by printing detailed information about the dataset structure and sample dimensions.
