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
│   ├── hyperparameter_tuning.py # 超参数调优脚本
│   └── generate_summary_table.py # 生成性能汇总表
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

## 最新功能

我们对系统进行了以下改进：

1. **改进的存储架构**：
   - 现在使用`results/task/model/experiment_id/`结构，其中`experiment_id`是基于参数哈希生成的唯一标识符
   - 相同参数的实验会覆盖而不是创建新的文件夹，便于进行多次尝试
   - 在每个模型目录下新增了`best_performance.json`文件，记录该模型在所有实验中的最佳性能

2. **自动超参数调优**：
   - 新增了`scripts/hyperparameter_tuning.py`脚本，支持三种搜索方法：
     - 网格搜索：系统地搜索所有参数组合
     - 随机搜索：在给定范围内随机采样参数组合
     - 贝叶斯优化：使用Optuna库实现的贝叶斯优化

3. **灵活的训练与评估流程**：
   - 分离了存储结果和生成汇总表的步骤，使训练更加灵活
   - 修改了`train_multi_model.py`，现在它只生成汇总结果JSON文件，不再自动生成汇总表
   - 修改后的流程允许您先进行多次实验或超参数调优，然后一次性生成所有结果的汇总表

4. **增强的结果汇总**：
   - 更新了`generate_summary_table.py`，它现在从每个模型的`best_performance.json`文件读取最佳性能
   - 不再需要手动选择哪些实验包含在汇总中，系统自动使用每个模型的最佳性能

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

## 超参数调优

系统提供了强大的超参数调优功能，可以帮助您找到最佳的模型参数配置：

```bash
# 使用Optuna贝叶斯优化进行超参数调优
python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer --search_method optuna --num_trials 20

# 使用网格搜索进行超参数调优
python scripts/hyperparameter_tuning.py --task_name HumanMotion --model_name lstm --search_method grid

# 使用随机搜索进行超参数调优
python scripts/hyperparameter_tuning.py --task_name HumanID --model_name vit --search_method random --num_trials 15
```

### 调优选项

- **搜索方法**：
  - `grid`：网格搜索，对所有参数组合进行穷举搜索（适用于较小的搜索空间）
  - `random`：随机搜索，从参数空间中随机采样（适用于较大的搜索空间）
  - `optuna`：贝叶斯优化，使用Optuna库实现（最高效，需要安装`pip install optuna`）

- **自定义参数范围**：
  
  您可以通过命令行参数定义自定义参数搜索范围：
  
  ```bash
  python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer \
      --search_method random --num_trials 20 \
      --learning_rates "0.001,0.0005,0.0001" \
      --batch_sizes "16,32,64" \
      --dropout_rates "0.1,0.3,0.5"
  ```
  
  对于随机搜索和贝叶斯优化，您也可以指定参数的连续范围：
  
  ```bash
  python scripts/hyperparameter_tuning.py --task_name MotionSourceRecognition --model_name transformer \
      --search_method optuna --num_trials 20 \
      --lr_min 0.0001 --lr_max 0.01 \
      --batch_size_min 16 --batch_size_max 128 \
      --dropout_min 0.0 --dropout_max 0.5
  ```

调优结果保存在每个模型和任务组合的`hyperparameter_tuning/`目录下，包括所有试验结果的详细CSV文件和找到的最佳参数的摘要JSON。

### 生成汇总表

要生成汇总不同模型和任务结果的表格，请运行：

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

这将在results目录中创建几个CSV表格：
- `full_summary.csv`：所有模型和任务的完整指标
- `accuracy_summary.csv`：专注于准确率指标
- `f1_score_summary.csv`：专注于F1分数指标
- `compact_summary.csv`：精简格式的关键指标
- `test_accuracy_pivot.csv`：不同模型和任务的测试准确率比较
- `test_f1_score_pivot.csv`：不同模型和任务的测试F1分数比较

### 多模型训练

您可以使用`train_multi_model.py`脚本一次训练多个模型：

```bash
# 在MotionSourceRecognition任务上训练多个模型
python train_multi_model.py --all_models "mlp lstm transformer vit" --task_name MotionSourceRecognition
```

完成后，您需要手动运行汇总表生成脚本：

```bash
python scripts/generate_summary_table.py --results_dir ./results
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Optuna (用于贝叶斯优化的超参数调优)

## SageMaker Integration

For large-scale training on AWS SageMaker, we provide a specialized runner that allows you to train models in the cloud. The SageMaker runner has been optimized to run multiple models on a single instance for each task, improving resource utilization and reducing costs.

### Key Features

- Run multiple models for each task on a single SageMaker instance
- Automated task-based batch processing
- Customizable instance types and training parameters
- Comprehensive job tracking and result summarization

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
