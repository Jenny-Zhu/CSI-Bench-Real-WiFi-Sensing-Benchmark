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

You can run the WiFi sensing benchmark on AWS SageMaker using the provided integration scripts. The SageMaker runner supports both local and cloud execution environments.

### Running on SageMaker

To run training jobs on SageMaker, use the `sagemaker_runner.py` script:

```bash
python sagemaker_runner.py --config configs/sagemaker_default_config.json
```

### Directory Structure

The SageMaker integration follows this workflow:

```
sagemaker_runner.py -> entry_script.py -> train_multi_model.py -> (uses TaskTrainer from) scripts/train_supervised.py
```

- `sagemaker_runner.py`: Creates and submits SageMaker training jobs
- `entry_script.py`: Entry point that runs in the SageMaker environment 
- `train_multi_model.py`: Handles training multiple models for a task
- `scripts/train_supervised.py`: Core training logic for single models

### Key Features

1. **Optimized Storage**:
   - Training output is automatically saved to the specified S3 bucket
   - Results are stored in a structured directory format: `s3://bucket-name/Benchmark_Log/TaskName/ModelName/`
   - Output includes model files, metrics, and evaluation results

2. **Debugging and Profiling**:
   - SageMaker Debugger and Profiler are disabled by default to save storage space
   - This prevents generation of unnecessary debug and profiling artifacts
   - Disabling these features significantly reduces the size of result files

3. **Batch Processing**:
   - The runner can process multiple tasks in a batch
   - Each task is run on a single instance with multiple models training sequentially

### Configuration Options

The SageMaker configuration file (`configs/sagemaker_default_config.json`) contains important settings:

```json
{
  "s3_data_base": "s3://your-bucket-name/Data/Benchmark/",
  "s3_output_base": "s3://your-bucket-name/Benchmark_Log/",
  "base_job_name": "wifi-sensing",
  "instance_type": "ml.g4dn.xlarge",
  "instance_count": 1,
  "max_run": 86400,
  "volume_size": 100,
  "available_tasks": ["MotionSourceRecognition", "HumanMotion", "HumanID"],
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
  "batch_wait_time": 30
}
```

### Example: Running Multiple Tasks

```bash
# Run all models for the MotionSourceRecognition task
python sagemaker_runner.py --config configs/sagemaker_default_config.json --task MotionSourceRecognition

# Run specific models for multiple tasks
python sagemaker_runner.py --config configs/sagemaker_default_config.json --tasks MotionSourceRecognition,HumanMotion --models lstm,transformer
```

### Troubleshooting SageMaker Jobs

If you encounter issues with storage or output files:

1. Check the CloudWatch logs for your SageMaker job
2. Verify the S3 paths in your configuration file
3. Ensure your IAM role has proper permissions for S3 access
4. Check if debug/profiler settings are properly disabled

## Enhanced Testing Capabilities

The framework has been enhanced to support testing models not only on in-distribution (ID) test data but also on various out-of-distribution scenarios:

- **Cross-Environment Testing**: Evaluate how models perform when deployed in new environments not seen during training
- **Cross-User Testing**: Measure model robustness when encountering new users not present in the training data
- **Cross-Device Testing**: Test model generalization to different devices or hardware configurations
- **Hard Cases**: Evaluate model performance on particularly challenging samples

### Running Extended Tests

To leverage these enhanced testing capabilities, you can use the provided scripts:

#### Supervised Learning
```bash
# Windows
run_extended_test_supervised.bat --model transformer --task MotionSourceRecognition

# Linux/Mac
./run_extended_test_supervised.sh --model transformer --task MotionSourceRecognition
```

#### Multitask Learning
```bash
# Windows
run_extended_test_multitask.bat --model transformer --tasks MotionSourceRecognition,HumanMotion

# Linux/Mac
./run_extended_test_multitask.sh --model transformer --tasks MotionSourceRecognition,HumanMotion
```

### Manual Control

You can also specify specific test splits to evaluate:

```bash
python scripts/train_supervised.py --model transformer --task_name MotionSourceRecognition --test_splits test_id,test_cross_env,test_cross_user

python scripts/train_multitask_adapter.py --model transformer --tasks MotionSourceRecognition,HumanMotion --test_splits test_id,test_cross_env
```

Use `--test_splits all` to automatically use all available test splits for each task.

## Results Analysis

The extended testing generates comprehensive reports for all test splits, including:

- Accuracy and F1-scores for each test split
- Confusion matrices for visualization
- JSON summary files with detailed metrics
- Comparative tables of performance across different test conditions

This provides deeper insights into model robustness and potential areas for improvement in real-world deployment scenarios.

## SageMaker Integration Updates

The SageMaker integration has been improved with the following enhancements:

### Parameter Handling Improvements

1. **Consistent Parameter Naming**: 
   - Parameters like `task_name` and `task` are now handled consistently across all scripts
   - Parameter names with hyphens (e.g., `--param-name`) are automatically converted to underscore format (`--param_name`)

2. **Enhanced Parameter Validation**:
   - Required parameters (`task_name`, `models`, `win_len`, `feature_size`) are validated before execution
   - Detailed error messages help identify missing parameters

3. **Improved SageMaker Environment Variables Support**:
   - SageMaker hyperparameters (`SM_HP_*`) are automatically detected and parsed
   - Path variables are correctly set based on SageMaker environment

4. **Default Value Alignment**:
   - Default values are now consistent between local and SageMaker environments
   - All model-specific defaults have been aligned to prevent inconsistencies

### Using the SageMaker Pipeline

To run training on SageMaker, use the `sagemaker_runner.py` script:

```bash
python sagemaker_runner.py --config configs/sagemaker_default_config.json
```

Or customize with specific parameters:

```bash
python sagemaker_runner.py --task MotionSourceRecognition --models transformer,vit --batch_size 32
```

When using SageMaker, ensure your data is organized correctly in the S3 bucket as described in the SageMaker Configuration section.

### Troubleshooting SageMaker Jobs

If you encounter issues with parameter passing:

1. Check the CloudWatch logs for detailed parameter handling information
2. Verify parameter names are consistent (use `task_name` instead of `task`)
3. Ensure required parameters are provided
4. Consider using the test environment mode with `--test-env` flag to verify setup

For more details on SageMaker usage, see the SageMaker Integration section below.
