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

- **Batch Processing**: Train all models for a task on a single instance, reducing resource costs
- **S3 Integration**: Automatically handles S3 paths for data input and output
- **Parameter Handling**: Converts hyperparameters to the format expected by the training script

### Parameter Passing in SageMaker

When using SageMaker, pay attention to the following parameter format requirements:

- **Use double dash (--) prefixes**: Parameters should use double dash prefixes (e.g., `--win-len`) not double underscore prefixes (`__win-len`).
- **Use dashes for parameter names**: In SageMaker environment, parameter names should use dashes instead of underscores (e.g., `win-len` instead of `win_len`).
- **Special parameter handling**: Some parameters like `task_name` are handled specially and should keep their underscore format.

Example of correct parameter format in SageMaker:
```python
hyperparameters = {
    'models': 'transformer,vit',
    'task_name': 'MotionSourceRecognition',  # This special parameter keeps underscores
    'win-len': 500,                          # Use dash instead of underscore
    'feature-size': 232,                     # Use dash instead of underscore
    'batch-size': 32                         # Use dash instead of underscore
}
```

Our `entry_script.py` contains validation logic to fix parameter format issues, but it's best to use the correct format from the start to avoid any potential problems.

### Running on SageMaker

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

### Known Issues and Solutions

#### Recent Fixes

1. **LabelMapper Attribute Error**
   - Fixed issue where the code was incorrectly trying to access `class_to_idx` attribute instead of `label_to_idx`
   - If you encounter attribute errors with label mapping, ensure you're using the correct attribute names

2. **Data Path Resolution**
   - The system now supports multiple methods to find the correct data path in SageMaker environment:
     - `--adaptive_path`: Automatically adapts to the SageMaker directory structure
     - `--try_all_paths`: Tries multiple combinations of paths to locate datasets
   - These flags are enabled by default in the SageMaker runner

3. **Enhanced Logging**
   - Improved debug logging to show detailed information about the SageMaker environment
   - Use `--debug` flag to enable verbose logging in SageMaker jobs

#### Best Practices for SageMaker Jobs

1. **Test Locally First**
   - Run a small-scale test locally using `local_runner.py` before submitting to SageMaker
   - Verify that your data loading pipeline works with the same dataset structure

2. **Validate File Paths**
   - Use the AWS CLI to verify your S3 paths before submitting jobs:
     ```bash
     aws s3 ls s3://rnd-sagemaker/Data/Benchmark/tasks/
     aws s3 ls s3://rnd-sagemaker/Data/Benchmark/tasks/MotionSourceRecognition/
     ```

3. **Use Debugging Flags**
   - Always include `--debug` flag in your SageMaker jobs for verbose logging
   - Enable `--adaptive_path` and `--try_all_paths` for robust path resolution

4. **Start Small**
   - Begin with a single task and model before scaling to multiple tasks
   - Use a smaller number of epochs for initial testing 

5. **Check Instance Types**
   - Ensure your SageMaker instance type has sufficient GPU memory for your models
   - `ml.g4dn.xlarge` works well for most models, but larger models may require `ml.g4dn.2xlarge`

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

## SageMaker Pipeline Enhancements

The SageMaker pipeline has been enhanced to ensure consistent results with the local pipeline. All training outputs, including visualizations, metrics, and model weights, are now properly saved and uploaded to S3.

## Key Improvements

1. **Complete Result Directory Structure**
   - Maintains the same directory structure as local pipeline: `/task_name/model_name/experiment_id/`
   - Saves best performance tracking for each model in `best_performance.json`
   - Creates proper experiment IDs for each combination of parameters

2. **Enhanced Visualizations**
   - Generates and saves learning curves (accuracy and loss)
   - Creates confusion matrices for each test split
   - Stores classification reports with detailed metrics
   - Preserves consistent file naming and formats with local pipeline

3. **Optimized S3 Upload**
   - Streamlined upload process to eliminate redundant directories
   - Removed timestamp and job ID from paths to prevent nested results
   - Skips debug files and other unnecessary outputs
   - Directly uploads task directory for cleaner S3 organization
   - Eliminates manifest files and other temporary artifacts

## S3 Path Cleanup

The updated pipeline addresses several issues with the previous S3 upload process:

1. **Fixed Path Structure**: No more redundant nested paths like `TaskName-timestamp/output/output/TaskName/`
2. **Debug File Removal**: Debug files like `debug-output/training_job_end.ts` are now automatically deleted
3. **No Manifest Files**: Removed upload manifest files that cluttered the S3 bucket

The optimized S3 path structure is now:

```
s3://your-bucket/Benchmark_Log/
└── task_name/
    ├── model_name/
    │   ├── best_performance.json
    │   └── experiment_id/
    │       ├── model_task_config.json
    │       ├── model_task_learning_curves.png
    │       ├── model_task_accuracy_curves.png
    │       ├── model_task_test_id_confusion.png
    │       └── ...
    └── multi_model_results.json
```

## Configuration Options

Several parameters are available to customize the SageMaker pipeline behavior:

```bash
# Basic options
--save_plots              # Save visualizations (enabled by default in SageMaker)
--save_confusion_matrix   # Save confusion matrices 
--save_learning_curves    # Save learning curves
--save_predictions        # Save model predictions
--save_model              # Save model weights

# Advanced options
--max_retries 3           # Maximum upload retry attempts
--plot_dpi 150            # DPI for saved plots
--plot_format png         # File format for plots (png, jpg, pdf, svg)
```

When running in a SageMaker environment, these visualization options are automatically enabled.
