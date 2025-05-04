# Meta-Learning Pipeline for WiFi Sensing

This README explains the fixes made to the meta-learning pipeline for the WiAL-Real-WiFi-Sensing-Benchmark and how to use the fixed code.

## Issues Fixed

1. **Null Byte Issues**: Fixed corrupt files with null bytes that prevented proper imports.
2. **Circular Import Problems**: Restructured imports to avoid circular dependencies.
3. **File Structure Expectations**: Created config files and directory structure to match script expectations.
4. **Sample ID Format Handling**: Improved sample ID handling to work with various formats.
5. **Empty Batch Handling**: Added robust error handling for empty batches or missing samples.
6. **Tensor Shape Compatibility**: Fixed tensor shape issues for cases with variable classes.

## Files Created

1. `fix_all_null_bytes.py` - Script to remove null bytes from Python files
2. `fix_imports.py` - Script to fix circular imports by setting up proper module structure
3. `fix_config_paths.py` - Script to create config files and directory structure
4. `meta_learning_data.py` - Standalone data loader module that avoids circular imports
5. `train_meta_standalone_fixed.py` - Fixed training script that handles errors gracefully
6. `run_adaptation_test.py` - Wrapper script for testing with proper structure

## How to Use

### Step 1: Fix Project Structure

Run the fix scripts in order:

```bash
python fix_all_null_bytes.py
python fix_imports.py
python fix_config_paths.py
```

### Step 2: Train Models

Train a meta-learning model using the fixed standalone script:

```bash
python train_meta_standalone_fixed.py --model_type mlp --num_iterations 5000 --save_dir results/meta_standalone --data_dir wifi_benchmark_dataset --task_name MotionSourceRecognition
```

Parameters:
- `--model_type`: Type of model to train (mlp, lstm, resnet18, transformer, vit)
- `--num_iterations`: Number of training iterations
- `--save_dir`: Directory to save results
- `--data_dir`: Path to the dataset
- `--task_name`: Task name (default: MotionSourceRecognition)
- `--n_way`: Number of classes per task (default: 3)
- `--k_shot`: Number of support examples per class (default: 5)
- `--q_query`: Number of query examples per class (default: 5)
- `--batch_size`: Number of tasks per batch (default: 1)
- `--inner_lr`: Inner loop learning rate (default: 0.01)
- `--meta_lr`: Outer loop learning rate (default: 0.001)

### Step 3: Test Adaptation

Test the trained model's adaptation capabilities:

```bash
python run_adaptation_test.py --model_type mlp --test_type adapt_5shot --data_dir wifi_benchmark_dataset --checkpoint results/meta_standalone/best_model.pth
```

Parameters:
- `--model_type`: Type of model to test
- `--test_type`: Type of test to run (test, cross_env, cross_user, cross_device, adapt_1shot, adapt_5shot, etc.)
- `--data_dir`: Path to the dataset
- `--checkpoint`: Path to the model checkpoint

## Understanding the Output

The training script will output:
- Training progress with loss and accuracy metrics
- Validation results
- Test results if specified

For debugging purposes, the script handles invalid tasks by:
- Skipping empty batches
- Using placeholder random data when no valid tasks are found
- Providing detailed logging of any issues

## Troubleshooting

If you encounter issues:
1. Check dataset paths and make sure they match the expected structure
2. Verify the meta-split files exist and are in the correct format
3. Increase logging verbosity if needed by changing the logging level
4. For memory issues on large datasets, reduce batch_size and num_workers

## Notes on Implementation

The implementation uses Model-Agnostic Meta-Learning (MAML) with:
- Inner loop adaptation using SGD
- Outer loop optimization using Adam
- Label remapping for handling variable class counts in tasks
- Support for multiple model architectures (MLP, LSTM, ResNet18, Transformer, ViT)

The code is designed to gracefully handle various error conditions that might occur with real-world WiFi sensing data. 