#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WiFi Sensing S3 Analyzer

A comprehensive tool for analyzing WiFi sensing experiments directly from S3.
This script combines functionality from multiple analysis scripts into a single tool.

Features:
- Direct reading from S3 without downloading files
- Summary tables of model performance across tasks
- Training/validation loss curves
- Performance comparisons and visualizations

Usage:
    python s3_wifi_analyzer.py --s3-dir s3://bucket/path/ --tasks FourClass ThreeClass --models Transformer ResNet18
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from io import StringIO, BytesIO
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s3_wifi_analyzer')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze WiFi sensing experiments directly from S3')
    parser.add_argument('--s3-dir', type=str, required=True,
                     help='S3 URI containing results (e.g., s3://bucket/path/)')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                     help='Local directory for analysis results (default: ./analysis_results)')
    parser.add_argument('--tasks', type=str, nargs='+',
                     help='Specific tasks to analyze (optional)')
    parser.add_argument('--models', type=str, nargs='+',
                     help='Specific models to analyze (optional)')
    parser.add_argument('--metrics', type=str, nargs='+',
                     default=['accuracy', 'precision', 'recall', 'f1', 'training_time'],
                     help='Metrics to analyze (default: accuracy, precision, recall, f1, training_time)')
    parser.add_argument('--plot-curves', action='store_true', default=True,
                     help='Plot training/validation loss curves (default: True)')
    parser.add_argument('--plot-comparisons', action='store_true', default=True,
                     help='Plot model performance comparisons (default: True)')
    parser.add_argument('--agg-method', type=str, choices=['mean', 'median', 'max'], default='mean',
                     help='Method to aggregate multiple runs (default: mean)')
    
    return parser.parse_args()

def parse_s3_uri(uri):
    """Parse S3 URI into bucket and prefix"""
    if uri.startswith('s3://'):
        uri = uri[5:]
    
    parts = uri.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    if prefix and not prefix.endswith('/'):
        prefix += '/'
        
    return bucket, prefix

def list_s3_directories(s3_client, bucket, prefix):
    """List directories in S3 bucket under given prefix"""
    paginator = s3_client.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    directories = []
    for page in result:
        if 'CommonPrefixes' in page:
            for obj in page['CommonPrefixes']:
                dir_path = obj['Prefix']
                # Get the directory name without the full path
                dir_name = dir_path[len(prefix):].rstrip('/')
                if dir_name:  # Skip empty strings
                    directories.append(dir_name)
    
    return directories

def read_s3_csv(s3_client, bucket, key):
    """Read CSV file from S3 into pandas DataFrame"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(content))
    except Exception as e:
        logger.error(f"Error reading S3 file {key}: {e}")
        return pd.DataFrame()

def read_s3_json(s3_client, bucket, key):
    """Read JSON file from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading S3 JSON file {key}: {e}")
        return {}

def find_result_files(s3_client, bucket, base_prefix, tasks=None, models=None):
    """Find all result files in S3 matching the criteria"""
    result_files = []
    
    # Determine tasks to analyze
    if tasks:
        task_prefixes = [f"{base_prefix}{task}/" for task in tasks]
    else:
        # List all task directories
        all_tasks = list_s3_directories(s3_client, bucket, base_prefix)
        task_prefixes = [f"{base_prefix}{task}/" for task in all_tasks]
    
    # For each task, find models
    for task_prefix in task_prefixes:
        task_name = task_prefix[len(base_prefix):].rstrip('/')
        
        if models:
            model_prefixes = [f"{task_prefix}{model}/" for model in models]
        else:
            # List all model directories for this task
            all_models = list_s3_directories(s3_client, bucket, task_prefix)
            model_prefixes = [f"{task_prefix}{model}/" for model in all_models]
        
        # For each model, find job directories
        for model_prefix in model_prefixes:
            model_name = model_prefix[len(task_prefix):].rstrip('/')
            
            # List job directories
            job_dirs = list_s3_directories(s3_client, bucket, model_prefix)
            
            for job_dir in job_dirs:
                job_prefix = f"{model_prefix}{job_dir}/"
                
                # Check if result files exist
                files_to_check = {
                    'classification_report': f"{job_prefix}classification_report.csv",
                    'training_results': f"{job_prefix}training_results.csv",
                    'hyperparams': f"{job_prefix}hyperparams.json"
                }
                
                # Alternate paths in output directory
                alt_prefix = f"{job_prefix}output/"
                alt_files = {
                    'classification_report': f"{alt_prefix}classification_report.csv",
                    'training_results': f"{alt_prefix}training_results.csv",
                    'hyperparams': f"{alt_prefix}hyperparams.json"
                }
                
                # Check for files
                found_files = {}
                
                # Try primary paths
                for file_type, file_path in files_to_check.items():
                    try:
                        s3_client.head_object(Bucket=bucket, Key=file_path)
                        found_files[file_type] = file_path
                    except ClientError:
                        # Try alternate path
                        try:
                            s3_client.head_object(Bucket=bucket, Key=alt_files[file_type])
                            found_files[file_type] = alt_files[file_type]
                        except ClientError:
                            # File not found at either location
                            pass
                
                # Only add the job if we found at least one required file
                if 'classification_report' in found_files or 'training_results' in found_files:
                    result_files.append({
                        'task': task_name,
                        'model': model_name,
                        'job': job_dir,
                        'job_dir': job_prefix,
                        'files': found_files
                    })
    
    return result_files

def extract_metrics(s3_client, bucket, result_files, target_metrics):
    """Extract metrics from S3 result files"""
    results_data = []
    
    for result in result_files:
        task = result['task']
        model = result['model']
        job = result['job']
        job_dir = result['job_dir']
        files = result['files']
        
        # Initialize metrics dictionary
        metrics = {
            'task': task,
            'model': model,
            'job': job
        }
        
        # Extract metrics from classification report
        if 'classification_report' in files:
            class_report_path = files['classification_report']
            class_df = read_s3_csv(s3_client, bucket, class_report_path)
            
            if not class_df.empty:
                # Extract macro avg metrics
                if 'macro avg' in class_df['class'].values:
                    macro_row = class_df[class_df['class'] == 'macro avg']
                    for metric in ['precision', 'recall', 'f1']:
                        if metric in macro_row.columns:
                            metrics[f'macro_{metric}'] = macro_row[metric].values[0]
                
                # Extract accuracy
                if 'accuracy' in class_df['class'].values:
                    acc_row = class_df[class_df['class'] == 'accuracy']
                    if 'precision' in acc_row.columns:
                        metrics['accuracy'] = acc_row['precision'].values[0]
        
        # Extract metrics from training results
        if 'training_results' in files:
            train_results_path = files['training_results']
            train_df = read_s3_csv(s3_client, bucket, train_results_path)
            
            if not train_df.empty:
                # Get last epoch values
                final_row = train_df.iloc[-1]
                
                # Training time
                if 'elapsed_time' in final_row:
                    metrics['training_time'] = final_row['elapsed_time']
                
                # Final metrics
                metrics_to_extract = [
                    'epoch', 'loss', 'accuracy', 
                    'val_loss', 'val_accuracy',
                    'val_precision', 'val_recall', 'val_f1'
                ]
                
                for metric in metrics_to_extract:
                    if metric in final_row:
                        metrics[f'final_{metric}'] = final_row[metric]
                
                # Also save the full training history for plots
                metrics['train_history'] = train_df
        
        # Extract hyperparameters
        if 'hyperparams' in files:
            hyperparams_path = files['hyperparams']
            hyperparams = read_s3_json(s3_client, bucket, hyperparams_path)
            
            if hyperparams:
                params_to_extract = [
                    'batch_size', 'learning_rate', 'num_epochs',
                    'weight_decay', 'model_name', 'win_len', 'feature_size'
                ]
                
                for param in params_to_extract:
                    if param in hyperparams:
                        metrics[f'param_{param}'] = hyperparams[param]
        
        # Add to results
        results_data.append(metrics)
    
    # Create DataFrame
    if results_data:
        results_df = pd.DataFrame(results_data)
        # Ensure all target metrics exist
        for metric in target_metrics:
            if metric not in results_df.columns:
                results_df[metric] = np.nan
        return results_df
    else:
        logger.error("没有找到有效的结果数据")
        return pd.DataFrame()

def aggregate_results(df, agg_method='mean'):
    """Aggregate multiple runs by task and model"""
    if df.empty:
        return df
    
    # Determine aggregation function
    if agg_method == 'mean':
        agg_func = np.mean
    elif agg_method == 'median':
        agg_func = np.median
    elif agg_method == 'max':
        agg_func = np.max
    else:
        agg_func = np.mean
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by task and model, then aggregate
    agg_df = df.groupby(['task', 'model'])[numeric_cols].agg(agg_func).reset_index()
    
    return agg_df

def create_summary_tables(df, output_dir):
    """Create summary tables for model performance"""
    if df.empty:
        return {}
    
    tables = {}
    
    # Overall summary table
    summary_cols = ['task', 'model', 'accuracy', 'macro_precision', 
                   'macro_recall', 'macro_f1', 'training_time']
    
    # Rename columns for clarity
    rename_map = {
        'macro_precision': 'precision',
        'macro_recall': 'recall',
        'macro_f1': 'f1'
    }
    
    summary_df = df.copy()
    for new_col, old_col in rename_map.items():
        if f'macro_{old_col}' in summary_df.columns:
            summary_df[new_col] = summary_df[f'macro_{old_col}']
    
    # Select only the columns we want
    summary_cols = [col for col in summary_cols if col in summary_df.columns]
    summary_df = summary_df[summary_cols]
    
    # Save overall summary
    overall_file = os.path.join(output_dir, "overall_summary.csv")
    summary_df.to_csv(overall_file, index=False)
    logger.info(f"保存总体摘要表到 {overall_file}")
    tables['overall'] = summary_df
    
    # Create per-task tables
    for task, task_df in df.groupby('task'):
        # Sort by accuracy
        if 'accuracy' in task_df.columns:
            task_df = task_df.sort_values('accuracy', ascending=False)
        
        # Save task summary
        task_file = os.path.join(output_dir, f"{task}_summary.csv")
        task_df.to_csv(task_file, index=False)
        logger.info(f"保存任务 {task} 摘要表到 {task_file}")
        tables[task] = task_df
    
    # Create per-model tables
    model_summaries = []
    for model in df['model'].unique():
        model_data = {'model': model}
        model_df = df[df['model'] == model]
        
        # Calculate average for each metric
        for col in summary_cols:
            if col not in ['task', 'model'] and col in model_df.columns:
                model_data[f'avg_{col}'] = model_df[col].mean()
        
        model_summaries.append(model_data)
    
    # Create DataFrame and sort by average accuracy
    if model_summaries:
        model_df = pd.DataFrame(model_summaries)
        if 'avg_accuracy' in model_df.columns:
            model_df = model_df.sort_values('avg_accuracy', ascending=False)
        
        # Save model summary
        model_file = os.path.join(output_dir, "model_summary.csv")
        model_df.to_csv(model_file, index=False)
        logger.info(f"保存模型摘要表到 {model_file}")
        tables['model'] = model_df
    
    return tables

def plot_learning_curves(df, output_dir):
    """Plot training and validation loss/accuracy curves for each experiment"""
    if df.empty:
        return
    
    # Create directory for learning curves
    curves_dir = os.path.join(output_dir, "learning_curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    # Group by task, model, and job
    for task in df['task'].unique():
        task_dir = os.path.join(curves_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        task_df = df[df['task'] == task]
        
        for model in task_df['model'].unique():
            model_dir = os.path.join(task_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            
            model_df = task_df[task_df['model'] == model]
            
            for idx, row in model_df.iterrows():
                if 'train_history' not in row:
                    continue
                
                job = row['job']
                train_history = row['train_history']
                
                # Plot loss curves
                plt.figure(figsize=(10, 6))
                plt.plot(train_history['epoch'], train_history['loss'], 'b-', label='Training Loss')
                
                if 'val_loss' in train_history.columns:
                    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Validation Loss')
                
                plt.title(f'Learning Curves - {task} - {model} - {job}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Save figure
                loss_file = os.path.join(model_dir, f"{job}_loss.png")
                plt.savefig(loss_file, dpi=300)
                plt.close()
                
                # Plot accuracy curves if available
                if 'accuracy' in train_history.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_history['epoch'], train_history['accuracy'], 'b-', label='Training Accuracy')
                    
                    if 'val_accuracy' in train_history.columns:
                        plt.plot(train_history['epoch'], train_history['val_accuracy'], 'r-', label='Validation Accuracy')
                    
                    plt.title(f'Accuracy Curves - {task} - {model} - {job}')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save figure
                    acc_file = os.path.join(model_dir, f"{job}_accuracy.png")
                    plt.savefig(acc_file, dpi=300)
                    plt.close()
                
                logger.info(f"保存学习曲线 - {task}/{model}/{job}")

def plot_model_comparisons(tables, output_dir):
    """Plot model performance comparisons"""
    if not tables:
        return
    
    # Create directory for comparison plots
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Plot overall model performance
    if 'model' in tables and not tables['model'].empty:
        model_df = tables['model']
        
        # Metrics to plot
        metrics = [col for col in model_df.columns if col.startswith('avg_') and not model_df[col].isnull().all()]
        
        if metrics:
            # Rename metrics for display
            plot_df = model_df.copy()
            plot_df.columns = [col.replace('avg_', '') if col.startswith('avg_') else col for col in plot_df.columns]
            
            # Set up the figure
            plt.figure(figsize=(12, 8))
            
            # Select metrics to plot
            plot_metrics = [m.replace('avg_', '') for m in metrics]
            plot_data = plot_df.set_index('model')[plot_metrics]
            
            # Create bar chart
            ax = plot_data.plot(kind='bar', figsize=(12, 6))
            plt.title('Average Model Performance Across All Tasks')
            plt.ylabel('Score')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.tight_layout()
            
            # Save figure
            model_file = os.path.join(comp_dir, "model_comparison.png")
            plt.savefig(model_file, dpi=300)
            plt.close()
            logger.info(f"保存模型比较图到 {model_file}")
    
    # Per-task model comparison (accuracy)
    if 'overall' in tables and not tables['overall'].empty:
        df = tables['overall']
        
        # Check if accuracy column exists
        if 'accuracy' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create pivot table for task-model accuracy
            pivot_df = df.pivot(index='model', columns='task', values='accuracy')
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
            plt.title('Model Accuracy by Task')
            plt.tight_layout()
            
            # Save figure
            acc_file = os.path.join(comp_dir, "accuracy_heatmap.png")
            plt.savefig(acc_file, dpi=300)
            plt.close()
            logger.info(f"保存准确率热图到 {acc_file}")
            
            # Create grouped bar chart
            plt.figure(figsize=(12, 8))
            pivot_df.plot(kind='bar', figsize=(12, 6))
            plt.title('Model Accuracy by Task')
            plt.ylabel('Accuracy')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.legend(title='Task')
            plt.tight_layout()
            
            # Save figure
            bar_file = os.path.join(comp_dir, "accuracy_by_task.png")
            plt.savefig(bar_file, dpi=300)
            plt.close()
            logger.info(f"保存任务准确率条形图到 {bar_file}")

def generate_summary_report(tables, output_dir):
    """Generate summary report in markdown format"""
    report_file = os.path.join(output_dir, "summary_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# WiFi Sensing Experiment Results Summary\n\n")
        f.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write model summary
        if 'model' in tables and not tables['model'].empty:
            f.write("## Model Performance Summary\n\n")
            f.write(tables['model'].to_markdown(index=False))
            f.write("\n\n")
        
        # Write task summaries
        f.write("## Task-specific Results\n\n")
        for task, df in tables.items():
            if task not in ['overall', 'model'] and not df.empty:
                f.write(f"### {task}\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n")
        
        # Add links to visualization directories
        f.write("## Visualizations\n\n")
        f.write("### Learning Curves\n\n")
        f.write("Learning curves showing training and validation metrics over epochs are available in the `learning_curves` directory.\n\n")
        
        f.write("### Model Comparisons\n\n")
        f.write("Model comparison visualizations are available in the `comparisons` directory.\n\n")
    
    logger.info(f"生成摘要报告: {report_file}")
    return report_file

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse S3 URI
    try:
        bucket, prefix = parse_s3_uri(args.s3_dir)
        logger.info(f"分析S3中的结果: s3://{bucket}/{prefix}")
    except Exception as e:
        logger.error(f"无效的S3 URI: {args.s3_dir}")
        sys.exit(1)
    
    # Create S3 client
    try:
        s3_client = boto3.client('s3')
    except Exception as e:
        logger.error(f"创建S3客户端出错: {e}")
        sys.exit(1)
    
    # Find result files
    logger.info("搜索结果文件...")
    result_files = find_result_files(s3_client, bucket, prefix, args.tasks, args.models)
    logger.info(f"找到 {len(result_files)} 个结果文件")
    
    if not result_files:
        logger.error(f"在 {args.s3_dir} 中没有找到结果文件")
        sys.exit(1)
    
    # Extract metrics
    logger.info("提取性能指标...")
    results_df = extract_metrics(s3_client, bucket, result_files, args.metrics)
    
    if results_df.empty:
        logger.error("提取有效指标失败")
        sys.exit(1)
    
    # Save raw results
    raw_file = os.path.join(args.output_dir, "raw_results.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"保存原始结果到 {raw_file}")
    
    # Aggregate results
    logger.info(f"使用{args.agg_method}方法聚合结果...")
    agg_df = aggregate_results(results_df, args.agg_method)
    
    # Create summary tables
    logger.info("创建汇总表...")
    tables = create_summary_tables(agg_df, args.output_dir)
    
    # Plot learning curves
    if args.plot_curves:
        logger.info("绘制学习曲线...")
        plot_learning_curves(results_df, args.output_dir)
    
    # Plot model comparisons
    if args.plot_comparisons:
        logger.info("绘制模型比较图...")
        plot_model_comparisons(tables, args.output_dir)
    
    # Generate summary report
    logger.info("生成摘要报告...")
    report_file = generate_summary_report(tables, args.output_dir)
    
    logger.info(f"分析完成! 结果保存到 {args.output_dir}")
    logger.info(f"摘要报告: {report_file}")

if __name__ == "__main__":
    main() 