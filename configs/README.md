# Model Configuration Files

This directory contains configuration files for training different models using the `local_runner.py` script.

## Available Model Configurations

1. **MLP (Multi-Layer Perceptron)**: `mlp_config.json`
2. **LSTM (Long Short-Term Memory)**: `lstm_config.json`
3. **ResNet-18**: `resnet18_config.json`
4. **Transformer**: `transformer_config.json`
5. **ViT (Vision Transformer)**: `vit_config.json`

## Usage

To train a model using the configuration files, use the `local_runner.py` script with the `--pipeline supervised` flag and specify the config file:

```bash
python local_runner.py --pipeline supervised --config_file [config_file_name]
```

For example:

```bash
# Train MLP model
python local_runner.py --pipeline supervised --config_file mlp_config.json

# Train LSTM model
python local_runner.py --pipeline supervised --config_file lstm_config.json

# Train ResNet-18 model
python local_runner.py --pipeline supervised --config_file resnet18_config.json

# Train Transformer model
python local_runner.py --pipeline supervised --config_file transformer_config.json

# Train ViT model
python local_runner.py --pipeline supervised --config_file vit_config.json
```

## Configuration Parameters

Each configuration file includes the following common parameters:

- `model_name`: The type of model to train
- `task`: The task to train on (e.g., "MotionSourceRecognition")
- `batch_size`: Batch size for training
- `num_epochs`: Number of epochs to train
- `learning_rate`: Learning rate for optimization
- `weight_decay`: Weight decay for regularization
- `win_len`: Window length for WiFi CSI data
- `feature_size`: Feature size for WiFi CSI data
- `results_subdir`: Subdirectory for saving results
- `training_dir`: Directory containing training data
- `output_dir`: Directory to save output results

Some models have additional parameters:

- **Transformer & ViT**:
  - `emb_dim`: Embedding dimension
  - `dropout`: Dropout rate

- **ResNet-18 & ViT**:
  - `in_channels`: Number of input channels

## Modifying Configurations

You can modify any parameter in the configuration files to adjust the training process. After modifying a configuration file, run the training with the updated file. 