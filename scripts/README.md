# Model Training Wrapper

`run_model.py` is a convenient wrapper script that automates the process of selecting and running appropriate model configurations based on the model type and task.

## Features

- **Automatic Config Selection**: Simply specify the model name and task
- **Config Generation**: Automatically creates appropriate config files for new model/task combinations
- **Parameter Override**: Allows overriding key parameters like epochs and batch size directly from the command line
- **Custom Config Support**: Optionally use your own custom config file

## Usage

### Basic Usage

Train a model by specifying the model architecture and task:

```bash
python run_model.py --model [model_name] --task [task_name]
```

Example:
```bash
python run_model.py --model lstm --task MotionSourceRecognition
```

### Available Models

- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer

### Common Tasks

- `MotionSourceRecognition`
- `HumanMotion`
- `DetectionandClassification`
- `HumanID`
- `NTUHAR`

### Overriding Parameters

You can override specific parameters from the command line:

```bash
python run_model.py --model lstm --task MotionSourceRecognition --epochs 20 --batch_size 64
```

### Using a Custom Config

You can also specify a custom config file to use:

```bash
python run_model.py --model lstm --task MotionSourceRecognition --custom_config my_custom_config.json
```

## How It Works

1. The script first checks if a configuration file exists for the specified model and task combination
2. If it exists, it uses that configuration
3. If not, it creates a new configuration with appropriate default parameters for the model
4. It applies any command line overrides to the configuration
5. Finally, it runs the `local_runner.py` script with the selected/generated configuration

## Output Structure

Results are saved in:
```
./results/[model]_[task]/
```

For example, LSTM model results for MotionSourceRecognition would be saved in:
```
./results/lstm_motionsourcerecognition/
``` 