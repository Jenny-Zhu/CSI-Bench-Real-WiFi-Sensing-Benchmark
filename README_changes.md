# WiFi Sensing Benchmark Project Improvements

This document summarizes the improvements we have made to the WiFi Sensing Benchmark project, focusing mainly on configuration file management and results storage mechanisms.

## Main Improvements

1. **Unified Directory Structure**
   - Moved configuration files from configs directory to results directory
   - Used a unified directory structure: `results/TASK/MODEL/EXPERIMENT_ID/`
   - Supervised learning and multitask learning use the same directory structure

2. **Best Model Selection Criteria**
   - Changed from using test accuracy to validation accuracy for determining the best model
   - Updated related code in `train_supervised.py` and `train_multitask_adapter.py`

3. **Improved Test Results Storage Format**
   - Used dictionary format to store results from multiple test sets: `{"test_set_1": test_acc_1, "test_set_2": test_acc_2}`
   - Added calculation and storage of F1 scores
   - Ensured consistent result formats between different learning methods

## Detailed Changes

### 1. local_runner.py
- Modified the `save_config` function to ensure configuration files are saved to the results directory
- Modified the `get_multitask_config` function to ensure it provides the correct task name
- Ensured multitask learning uses a single task name as the directory name

### 2. train_supervised.py
- Updated the code to use validation accuracy instead of test accuracy to determine the best model
- Improved the results storage format, using dictionary format to store test accuracy and F1 scores

### 3. train_multitask_adapter.py
- Modified the `evaluate` function, adding F1 score calculation capability
- Updated places where the evaluate function is used, ensuring F1 scores are correctly used
- Improved the test results storage format, using dictionary format to store test accuracy and F1 scores
- Ensured the use of validation accuracy instead of test accuracy to determine the best model

## Effects

These changes make the project's configuration and results management more consistent and clear, facilitating comparison of results between different experiments. Now:

1. All results and configuration files are uniformly stored in the results directory, using a consistent directory structure
2. The best model selection criterion is uniformly based on validation accuracy, which is more consistent with actual machine learning practices
3. Test results storage format is standardized to dictionary format, making it easier to handle results from multiple test sets
4. F1 scores are added, providing more comprehensive metrics for evaluation

# SageMaker Runtime Environment Improvements

We have made the following improvements to the SageMaker runtime environment:

## 1. Parameter Format Standardization

- Modified the parameter names in the hyperparameters dictionary in `sagemaker_runner.py`, changing `task` to `task_name` to be consistent with parameter names in `train_multi_model.py`
- Added logic in `sagemaker_runner.py` to convert parameter names with dashes to underscore format in model_params, ensuring consistency with Python command-line parameter specifications
- Added parameter conversion logic in `entry_script.py` to handle parameters passed by SageMaker, converting dash format to underscore format
- Added the same parameter format conversion logic in the generated wrapper_script, ensuring parameters can be processed uniformly regardless of how they are run

## 2. Environment Testing Improvements

- Enhanced the `test_sagemaker_env.py` script, adding parameter format conversion testing functionality
- Added comprehensive environment assessment reporting, including PyTorch, Horovod, SMDebug status, and parameter format conversion test results
- Provided more detailed report output to help diagnose potential environment issues

## 3. Configuration Save Location Standardization

Maintained consistency in configuration save locations between local runs and SageMaker runs, ensuring configuration files are saved in the same directory as model results.

These improvements ensure consistency and correctness of parameter passing when running deep learning tasks in the SageMaker environment, solving problems previously caused by inconsistent parameter name formats. They also provide better environment diagnostic tools to help quickly identify and resolve potential environment configuration issues.

# SageMaker Runner Simplification

We have simplified the SageMaker runner to make it easier to use, similar to the local runner:

## 1. Simplified Parameter Handling

- Removed complex command-line parameter parsing, changing to only support reading parameters from a JSON configuration file
- Added a new `run_from_config` function as the main entry point, similar to local_runner.py
- Simplified command-line parameters to only need a `--config` parameter to specify the configuration file path

## 2. Streamlined Default Configuration

- Cleaned up `configs/sagemaker_default_config.json`, removing unnecessary parameters
- Removed redundant parameters such as `mode`, `task_class_mapping`, etc.
- Maintained consistency with the `local_default_config.json` format

## 3. Improved Interface

- Changed the entry_point to uniformly use `entry_script.py` instead of `sagemaker_entry_point.py`
- Simplified the initialization method of the SageMakerRunner class, directly accepting a configuration object
- Added better parameter type handling, ensuring task and model parameters are always passed in the appropriate format

## 4. Usage Examples

```python
# Run from command line
python sagemaker_runner.py --config configs/my_custom_config.json

# Or use in code
from sagemaker_runner import run_from_config
run_from_config('configs/my_custom_config.json')
```

These improvements make the SageMaker runner more similar to the local runner in functionality and style, making it easier to use similar code and configuration methods in different environments.

# Data Path Optimization

To improve SageMaker training efficiency and reduce unnecessary data downloads, we have optimized the data processing logic:

## 1. Task-Specific Data Paths

- Modified the `_prepare_inputs` method to ensure each training job only downloads data specific to that task
- Data paths now in the format `s3://rnd-sagemaker/Data/Benchmark/tasks/{TASK_NAME}/`
- This avoids downloading the entire dataset (over 400GB) to each training instance

## 2. Data Path Optimization for Multitask Learning

- Improved the `_prepare_multitask_inputs` method for multitask learning
- When there is only one task, directly use that task's path
- When there are multiple tasks, use the parent directory path of the tasks, and provide detailed log information

## 3. Increased EBS Volume Size

- Increased default EBS volume size from 30GB to 500GB to accommodate the 400+GB dataset
- Based on actual usage, users can further adjust this parameter in the configuration file as needed

These optimizations ensure SageMaker only downloads necessary data to the training instance, reducing data transfer time and storage costs. This is particularly meaningful for large datasets and can significantly improve training efficiency.

# Instance Type and Storage Space Optimization

To solve SageMaker instance storage limitation issues, we have made the following improvements:

## 1. Upgraded Instance Type

- Upgraded the default instance type from `ml.g4dn.xlarge` to `ml.g4dn.2xlarge`
- The 2xlarge instance provides greater computing power and storage space, more suitable for handling large datasets

## 2. Removed Custom EBS Volume Size Setting

- Removed the `ebs_volume_size` parameter setting, letting SageMaker use the default storage capacity of the instance type
- Avoided errors caused by attempting to set storage space exceeding the maximum supported by the instance
- ml.g4dn type instances have local instance storage, eliminating the need to over-configure EBS volumes

## 3. Data Path Optimization Still Effective

- Previous data path optimization remains effective, ensuring only task-specific data is downloaded
- By only downloading needed data, the demand for large-capacity storage is reduced
- This is particularly important for the limited storage space of g4dn instances

These changes resolved the "ValidationException: Invalid VolumeSizeInGB" error, ensuring SageMaker training jobs can start successfully, while optimizing data paths to reduce the actual storage space needed. 