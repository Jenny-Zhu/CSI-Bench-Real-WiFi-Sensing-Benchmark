import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Define the base directory where results are stored
base_dir = "/Users/leo/Desktop/benchmark_result"
results_data = []

# Function to extract data from best_performance.json
def extract_performance_data(json_path, pipeline, task_name, model_name):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Create a row dictionary with basic information
        row = {
            'pipeline': pipeline,
            'task_name': task_name,
            'model_name': model_name,
            'experiment_id': data.get('experiment_id', ''),
            'avg_test_accuracy': data.get('avg_test_accuracy', ''),
            'best_epoch': data.get('best_epoch', '')
        }
        
        # Extract test metrics for each test type
        test_metrics = data.get('test_metrics', {})
        for test_name, metrics in test_metrics.items():
            for metric_name, value in metrics.items():
                column_name = f"{test_name}_{metric_name}"
                row[column_name] = value
                
        return row
    except Exception as e:
        print(f"处理{json_path}时出错: {e}")
        return None

# Traverse the directory structure and collect data
for pipeline_dir in os.listdir(base_dir):
    pipeline_path = os.path.join(base_dir, pipeline_dir)
    if not os.path.isdir(pipeline_path) or pipeline_dir.startswith('.'):
        continue
        
    for task_dir in os.listdir(pipeline_path):
        task_path = os.path.join(pipeline_path, task_dir)
        if not os.path.isdir(task_path) or task_dir.startswith('.'):
            continue
            
        for model_dir in os.listdir(task_path):
            model_path = os.path.join(task_path, model_dir)
            if not os.path.isdir(model_path) or model_dir.startswith('.'):
                continue
                
            # Check for best_performance.json
            json_path = os.path.join(model_path, 'best_performance.json')
            if os.path.exists(json_path):
                print(f"处理文件 {json_path}")
                row_data = extract_performance_data(json_path, pipeline_dir, task_dir, model_dir)
                if row_data:
                    results_data.append(row_data)
                    
# Directly create DataFrame, ensuring no duplicate data
if results_data:
    # Get all keys
    all_keys = set()
    for row in results_data:
        all_keys.update(row.keys())
    
    # Create normalized data, ensuring each row has the same columns
    normalized_data = []
    for row in results_data:
        normalized_row = {}
        for key in all_keys:
            normalized_row[key] = row.get(key, np.nan)
        normalized_data.append(normalized_row)
    
    # Create DataFrame from normalized data
    results_df = pd.DataFrame(normalized_data)
else:
    results_df = pd.DataFrame()

# Confirm no duplicate columns
if not results_df.empty:
    # Check for duplicate column names
    if results_df.columns.duplicated().any():
        print("发现重复列名，正在清理...")
        # Keep only the first occurrence
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        print(f"清理后的列数: {len(results_df.columns)}")

# Get all unique test metrics
metric_columns = [col for col in results_df.columns if '_' in col]
test_types = sorted(set([col.split('_')[0] for col in metric_columns]))
metric_types = sorted(set([col.split('_', 1)[1] for col in metric_columns]))

print(f"找到 {len(test_types)} 种测试类型: {test_types}")
print(f"找到 {len(metric_types)} 种指标类型: {metric_types}")

# Create a more organized and readable table
basic_columns = ['pipeline', 'task_name', 'model_name', 'experiment_id', 'avg_test_accuracy', 'best_epoch']
organized_columns = []

# Ensure basic columns exist in DataFrame before adding
for col in basic_columns:
    if col in results_df.columns:
        organized_columns.append(col)

# Organize metrics by test_type
for test_type in test_types:
    for metric in metric_types:
        col_name = f"{test_type}_{metric}"
        if col_name in results_df.columns:
            organized_columns.append(col_name)
            
# Reorder the DataFrame columns
organized_df = results_df[organized_columns]

# Display basic info
print("\nDataFrame形状:", organized_df.shape)
print("\n列名:", organized_df.columns.tolist())

# Save the results to CSV
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results_summary.csv')
organized_df.to_csv(csv_path, index=False)
print(f"\n结果已保存到 '{csv_path}'")

# Print the first few rows of the DataFrame
print("\nDataFrame样本:")
print(organized_df.head()) 