import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

# Configuration - You can modify these variables
data_dir = r"C:\Users\weiha\Desktop\benchmark_result"  # Base directory containing results
pipeline_list = ["supervised"]  # List of pipelines to analyze
task_name_list = ["BreathingDetection", "BreathingDetection_Subset"]  # List of tasks to analyze
model_name_list = ["lstm", "patchtst"]  # List of models to analyze

# Initialize an empty list to store results
results_data = []

# Iterate through all combinations
for pipeline in pipeline_list:
    for task_name in task_name_list:
        for model_name in model_name_list:
            # Find all experiment folders for the current combination
            exp_pattern = os.path.join(data_dir, pipeline, task_name, model_name, "params_*")
            exp_folders = glob(exp_pattern)
            
            print(f"Found {len(exp_folders)} experiments for {pipeline}/{task_name}/{model_name}")
            
            # Process each experiment
            for exp_folder in tqdm(exp_folders, desc=f"Processing {pipeline}/{task_name}/{model_name}"):
                # Extract experiment ID
                exp_id = os.path.basename(exp_folder)
                
                # Define paths for config and results files
                config_filename = f"{model_name}_{task_name}_config.json"
                results_filename = f"{model_name}_{task_name}_results.json"
                
                config_path = os.path.join(exp_folder, config_filename)
                results_path = os.path.join(exp_folder, results_filename)
                
                # Skip if either file doesn't exist
                if not os.path.exists(config_path) or not os.path.exists(results_path):
                    print(f"Missing files for {exp_folder}, skipping")
                    continue
                
                # Read config file
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Extract required fields from config
                    learning_rate = config_data.get('learning_rate')
                    weight_decay = config_data.get('weight_decay')
                    seed = config_data.get('seed')
                    
                    # Skip if seed is not available (old version)
                    if seed is None:
                        print(f"Seed not available in {exp_folder}, skipping")
                        continue
                        
                except Exception as e:
                    print(f"Error reading config file {config_path}: {e}")
                    continue
                
                # Read results file
                try:
                    with open(results_path, 'r') as f:
                        results_data_json = json.load(f)
                except Exception as e:
                    print(f"Error reading results file {results_path}: {e}")
                    continue
                
                # Create a dictionary for the current experiment
                exp_result = {
                    'pipeline': pipeline,
                    'task_name': task_name,
                    'model_name': model_name,
                    'experiment_id': exp_id,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'seed': seed,
                }
                
                # Extract test results for all test sets
                for test_name, test_results in results_data_json.items():
                    if isinstance(test_results, dict):
                        for metric_name, metric_value in test_results.items():
                            column_name = f"{test_name}_{metric_name}"
                            exp_result[column_name] = metric_value
                
                # Add to results list
                results_data.append(exp_result)

# Create DataFrame from results
if results_data:
    results_df = pd.DataFrame(results_data)
    
    # Display summary
    print("\nResults Summary:")
    print(f"Total experiments: {len(results_df)}")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Calculate average metrics by model and task
    print("\nAverage metrics by model and task:")
    for task in task_name_list:
        for model in model_name_list:
            task_model_df = results_df[(results_df['task_name'] == task) & (results_df['model_name'] == model)]
            if not task_model_df.empty:
                print(f"\n{task} - {model} (count: {len(task_model_df)})")
                
                # Find metrics columns
                metric_columns = [col for col in task_model_df.columns if any(col.endswith(m) for m in ['_loss', '_accuracy', '_f1'])]
                
                if metric_columns:
                    avg_metrics = task_model_df[metric_columns].mean()
                    for metric, value in avg_metrics.items():
                        print(f"  {metric}: {value:.4f}")
    
    # Save results to CSV
    output_path = os.path.join("result_analysis", "all_results_summary.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Optional: Display the DataFrame
    print("\nFirst few rows of the DataFrame:")
    print(results_df.head())
else:
    print("No results found matching the criteria.") 