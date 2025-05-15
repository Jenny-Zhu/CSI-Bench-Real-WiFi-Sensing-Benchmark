# WiFi Sensing Benchmark

A comprehensive benchmark and training system for WiFi sensing using CSI data.

## Overview

This repository provides a unified framework for training and evaluating deep learning models on WiFi Channel State Information (CSI) data for various sensing tasks. The framework supports both local execution and cloud-based training on AWS SageMaker.

## Installation and Setup

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended, but not required. Apple silicon is also accepted, but need to install Pytorch specifically. See https://developer.apple.com/metal/pytorch/ for more details about install pytorch for apple silicon)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/WiAL-Real-WiFi-Sensing-Benchmark.git
   cd WiAL-Real-WiFi-Sensing-Benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Data Download:

   Our full dataset will be released to public by camera ready deadline of NeurIPS 2025. For reviewers, please use the link in the paper manuscript to access our data through Kaggle. After downloaded the dataset, it should be in the following :
    
  ```
  CSI-Bench/
  ├── HumanActivityRecognition/
  ├── FallDetection/
  ├── BreathingDetection/
  ├── Localization/
  ├── HumanIdentification/
  ├── MotionSourceRecognition/
  └── ProximityRecognition/
  ```
  Each task directory follows a consistent structure:
  ```
  TaskName/
  ├── sub_Human/                    # Contains all user data
  │   ├── user_U01/                 # Data for specific user
  │   │   ├── act_ActivityName/     # Data for specific activity
  │   │   │   ├── env_E01/          # Data from specific environment
  │   │   │   │   ├── device_DeviceName/  # Data from specific device
  │   │   │   │   │   └── session_TIMESTAMP__freqFREQ.h5  # Individual CSI recordings
  │   ├── user_U02/
  │   └── ...
  ├── metadata/                     # Metadata for the task
  │   ├── sample_metadata.csv       # Detailed information about each sample
  │   └── label_mapping.json        # Maps activity labels to indices
  └── splits/                       # Dataset splits for experiments
      ├── train_id.json             # Training set IDs
      ├── val_id.json               # Validation set IDs
      ├── test_id.json              # Test set IDs
      ├── test_easy.json            # Easy difficulty test set
      ├── test_medium.json          # Medium difficulty test set
      └── test_hard.json            # Hard difficulty test set


## Local Execution

The main entry point for local execution is `scripts/local_runner.py`. This script handles configuration loading, model training, and result storage.

### Configuration

Edit the local configuration file at `configs/local_default_config.json` to set your data path and other parameters:

```json
{
  "pipeline": "supervised",
  "training_dir": "/path/to/your/data/",
  "output_dir": "./results", 
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"],
  "task": "YourTask",
  "win_len": 500,
  "feature_size": 232,
  "batch_size": 32,
  "epochs": 100,
  "test_splits": "all"
}
```

Key parameters:
- `pipeline`: Training pipeline type (only have `supervised` for now)
- `training_dir`: Path to your data directory. 
- `output_dir`: Directory to save results (default: `./results`)
- `available_models`: Model types to train, default list is all models in this project
- `task`: Task name (see Available Tasks)
- `batch_size`, `epochs`: Training parameters

### Running Models

Basic usage:
```bash
python scripts/local_runner.py
```

### Available Models

- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer
- `patchtst`: PatchTST (Patch Time Series Transformer)
- `timesformer1d`: TimesFormer for 1D signals

### Available Tasks (Make sure you downloaded the whole dataset for corresponding task)

- `MotionSourceRecognition`
- `BreathingDetection_Subset`
- `Localization`
- `FallDetection`
- `ProximityRecognition`
- `HumanActivityRecognition`
- `HumanIdentification`




## Results Organization

Training results are saved with the following structure:

```
results/
├── task_name/                 # Name of the task
│   ├── model_name/            # Name of the model
│   │   ├── best_performance.json     # Record of best performance
│   │   ├── params_hash/              # Experiment identifier
│   │   │   ├── model_task_config.json           # Model configuration
│   │   │   ├── model_task_results.json          # Training metrics
│   │   │   ├── model_task_summary.json          # Performance summary
│   │   │   ├── model_task_test_confusion.png    # Confusion matrix
│   │   │   ├── classification_report_test.csv   # Classification metrics
│   │   │   └── checkpoint/                      # Saved model weights
│   │   └── 
│   |
│   └── 
└── ...
```

## Advanced Features

### Multi-Task Learning

If you want to do multi-task learning, please switch branch to 


## SageMaker Integration 

The repository also supports training on AWS SageMaker.



```

## Citation

If you use this code in your research, please cite:
```
@article{wifi_sensing_benchmark,
  title={WiFi Sensing Benchmark: A Comprehensive Evaluation Framework for WiFi Sensing},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
