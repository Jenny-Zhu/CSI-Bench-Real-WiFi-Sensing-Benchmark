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
│   ├── run_model.py          # Main model training wrapper
│   ├── local_runner.py       # Execution runner script
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
└── train.py                  # Training shortcut script
```

## Training Models

You can easily train different models using our training shortcut script:

```bash
python train.py --model [model_name] --task [task_name]
```

Or use the full path to the wrapper script:

```bash
python scripts/run_model.py --model [model_name] --task [task_name]
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

### Example

```bash
# Train an LSTM model for MotionSourceRecognition
python train.py --model lstm --task MotionSourceRecognition

# Train a transformer model with custom parameters
python train.py --model transformer --task HumanID --epochs 20 --batch_size 64
```

## Configuration

Model configurations are stored in the `configs/` directory. Each model has its own configuration file with parameters optimized for that architecture.

You can directly modify these configs, or override key parameters from the command line:

```bash
python train.py --model lstm --task MotionSourceRecognition --epochs 20 --batch_size 64 --output_dir ./custom_results
```

## Results

Training results are saved in the `results/` directory, including:

- Trained model weights
- Training metrics and logs
- Confusion matrices
- Classification reports

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib, Seaborn
