#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Summary Table Script

This script generates a summary table of performance metrics across different models and tasks.
It collects results from the best_performance.json files in each model directory and creates 
consolidated CSV tables. This approach ensures that only the best performance for each model-task 
combination is included in the summary.

Usage:
    python scripts/generate_summary_table.py --results_dir ./results
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate summary tables of model performance")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing results (default: ./results)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the summary tables (default: same as results_dir)")
    parser.add_argument("--include_experiments", action="store_true", 
                        help="Whether to include individual experiment results")
    return parser.parse_args()

def find_best_performance_files(results_dir):
    """
    Find all best_performance.json files in model directories.
    
    Args:
        results_dir: Base directory containing results
        
    Returns:
        List of paths to best performance files
    """
    # Pattern matches best_performance.json files in task/model directories
    pattern = os.path.join(results_dir, "*", "*", "best_performance.json")
    return glob(pattern)

def find_summary_files(results_dir, include_experiments=False):
    """
    Find all summary files in the results directory structure.
    If include_experiments is True, also include individual experiment summary files.
    
    Args:
        results_dir: Base directory containing results
        include_experiments: Whether to include individual experiment summaries
        
    Returns:
        List of paths to summary files
    """
    summary_files = []
    
    # First, find all best_performance.json files for each model
    best_perf_files = find_best_performance_files(results_dir)
    summary_files.extend(best_perf_files)
    
    # If requested, also include individual experiment summary files
    if include_experiments:
        experiment_summary_pattern = os.path.join(results_dir, "**", "*_summary.json")
        experiment_files = glob(experiment_summary_pattern, recursive=True)
        # Filter out files that are not in experiment directories
        experiment_files = [f for f in experiment_files if "params_" in os.path.dirname(f)]
        summary_files.extend(experiment_files)
    
    return summary_files

def extract_metrics_from_file(file_path):
    """
    Extract metrics from a summary file.
    
    Args:
        file_path: Path to summary file
        
    Returns:
        Dictionary containing extracted metrics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a best_performance.json file or a regular summary file
        is_best_performance = "best_experiment_id" in data
        
        if is_best_performance:
            # Extract task and model from directory structure
            # Path format: results/task_name/model_name/best_performance.json
            dir_path = os.path.dirname(file_path)
            model_name = os.path.basename(dir_path)
            task_name = os.path.basename(os.path.dirname(dir_path))
            
            # Create a metrics dictionary with data from best_performance.json
            metrics = {
                'task_name': task_name,
                'model_name': model_name,
                'test_accuracy': data.get('best_test_accuracy', 0.0),
                'test_f1_score': data.get('best_test_f1_score', 0.0),
                'best_experiment_id': data.get('best_experiment_id', None),
                'best_experiment_params': data.get('best_experiment_params', {})
            }
            
            # Add experiment parameters as top-level keys for easier filtering
            if 'best_experiment_params' in data and isinstance(data['best_experiment_params'], dict):
                for param, value in data['best_experiment_params'].items():
                    metrics[f'param_{param}'] = value
        else:
            # Regular summary file, extract metrics directly
            metrics = data
            
            # Make sure task_name and model_name are present
            if 'task_name' not in metrics and 'task' in metrics:
                metrics['task_name'] = metrics['task']
            if 'model_name' not in metrics and 'model' in metrics:
                metrics['model_name'] = metrics['model']
        
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def create_summary_table(summary_files):
    """
    Create a summary table from all summary files.
    
    Args:
        summary_files: List of paths to summary files
        
    Returns:
        DataFrame containing the summary table
    """
    all_metrics = []
    
    for file_path in summary_files:
        metrics = extract_metrics_from_file(file_path)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics found")
        return None
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Define key columns for sorting
    sort_columns = ['task_name', 'model_name']
    
    # Sort by task and model
    if all(col in df.columns for col in sort_columns):
        df = df.sort_values(by=sort_columns)
    
    return df

def generate_specialized_tables(df):
    """
    Generate specialized tables based on different criteria.
    
    Args:
        df: DataFrame containing all metrics
        
    Returns:
        Dictionary of DataFrames with different specialized views
    """
    tables = {
        'full': df,  # Full table with all metrics
    }
    
    # Create accuracy table
    accuracy_cols = ['model_name', 'task_name'] + [col for col in df.columns if 'accuracy' in col]
    if all(col in df.columns for col in accuracy_cols[:2]) and len(accuracy_cols) > 2:
        tables['accuracy'] = df[accuracy_cols]
    
    # Create F1 score table
    f1_cols = ['model_name', 'task_name'] + [col for col in df.columns if 'f1_score' in col]
    if all(col in df.columns for col in f1_cols[:2]) and len(f1_cols) > 2:
        tables['f1_score'] = df[f1_cols]
    
    # Create compact table with only key metrics
    key_metrics = ['model_name', 'task_name', 'test_accuracy', 'test_f1_score', 'best_experiment_id']
    # Filter to columns that actually exist in the DataFrame
    key_metrics = [col for col in key_metrics if col in df.columns]
    if len(key_metrics) > 2:  # Only create if we have at least one metric beyond model and task
        tables['compact'] = df[key_metrics]
    
    # Create parameters table
    param_cols = ['model_name', 'task_name'] + [col for col in df.columns if col.startswith('param_')]
    if all(col in df.columns for col in param_cols[:2]) and len(param_cols) > 2:
        tables['parameters'] = df[param_cols]
    
    return tables

def create_pivot_tables(df):
    """
    Create pivot tables to compare models across tasks.
    
    Args:
        df: DataFrame containing all metrics
        
    Returns:
        Dictionary of DataFrames with pivot tables
    """
    pivot_tables = {}
    
    # Make sure we have the necessary columns
    req_cols = ['model_name', 'task_name']
    if not all(col in df.columns for col in req_cols):
        return pivot_tables
    
    # Identify metrics to pivot
    metric_cols = [col for col in df.columns 
                  if any(tag in col for tag in ['accuracy', 'f1_score']) 
                  and col not in req_cols]
    
    # Create a pivot table for each metric
    for metric in metric_cols:
        try:
            pivot = df.pivot_table(
                index='model_name',
                columns='task_name',
                values=metric,
                aggfunc='max'  # Take max value if there are duplicates
            )
            
            # Add a row with average performance across tasks
            pivot.loc['Average'] = pivot.mean()
            
            # Add a column with average performance across models
            pivot['Average'] = pivot.mean(axis=1)
            
            pivot_tables[metric] = pivot
        except Exception as e:
            print(f"Error creating pivot table for {metric}: {e}")
    
    return pivot_tables

def create_ranking_tables(df):
    """
    Create tables ranking models by performance on each task.
    
    Args:
        df: DataFrame containing all metrics
        
    Returns:
        Dictionary of DataFrames with ranking tables
    """
    ranking_tables = {}
    
    # Check if we have the necessary columns
    if not all(col in df.columns for col in ['model_name', 'task_name']):
        return ranking_tables
    
    # Metrics to rank by
    metrics = [col for col in df.columns if 'accuracy' in col or 'f1_score' in col]
    
    for metric in metrics:
        if metric in df.columns:
            try:
                # Group by task and rank models within each task
                ranking = df.sort_values(['task_name', metric], ascending=[True, False])
                
                # Add rank column
                ranking_with_rank = []
                for task, group in ranking.groupby('task_name'):
                    group = group.copy()
                    group['rank'] = range(1, len(group) + 1)
                    ranking_with_rank.append(group)
                
                if ranking_with_rank:
                    ranking_df = pd.concat(ranking_with_rank)
                    # Select relevant columns
                    cols = ['task_name', 'model_name', metric, 'rank']
                    ranking_tables[f"ranking_{metric}"] = ranking_df[cols]
            except Exception as e:
                print(f"Could not create ranking table for {metric}: {e}")
    
    return ranking_tables

def save_tables(tables, output_dir):
    """
    Save tables to CSV files.
    
    Args:
        tables: Dictionary of DataFrames to save
        output_dir: Directory to save the tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, table in tables.items():
        output_path = os.path.join(output_dir, f"{name}_summary.csv")
        table.to_csv(output_path, index=True)
        print(f"Saved {name} summary to {output_path}")

def main():
    """Main function to generate summary tables"""
    args = parse_args()
    
    # Use output_dir if specified, otherwise use results_dir
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all summary files
    print(f"Looking for model performance files in {args.results_dir}...")
    summary_files = find_summary_files(args.results_dir, args.include_experiments)
    
    if not summary_files:
        print("No summary files found. Make sure the results directory is correct.")
        return 1
    
    print(f"Found {len(summary_files)} model performance files.")
    
    # Create summary table
    summary_df = create_summary_table(summary_files)
    
    if summary_df is None:
        print("Failed to create summary table.")
        return 1
    
    # Save full summary table
    summary_csv_path = os.path.join(output_dir, "full_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved full summary table to {summary_csv_path}")
    
    # Generate specialized tables
    specialized_tables = generate_specialized_tables(summary_df)
    
    # Save each specialized table
    for name, table in specialized_tables.items():
        if name != 'full':  # Full table already saved
            table_path = os.path.join(output_dir, f"{name}_summary.csv")
            table.to_csv(table_path, index=False)
            print(f"Saved {name} summary table to {table_path}")
    
    # Create and save pivot tables
    pivot_tables = create_pivot_tables(summary_df)
    
    for metric, pivot in pivot_tables.items():
        pivot_path = os.path.join(output_dir, f"{metric}_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"Saved {metric} pivot table to {pivot_path}")
    
    # Create ranking tables
    print("Creating model ranking tables...")
    ranking_tables = create_ranking_tables(summary_df)
    
    # Combine all tables
    all_tables = {**specialized_tables, **pivot_tables, **ranking_tables}
    
    # Save tables
    print(f"Saving tables to {output_dir}...")
    save_tables(all_tables, output_dir)
    
    # Print some key statistics
    num_tasks = len(summary_df['task_name'].unique())
    num_models = len(summary_df['model_name'].unique())
    
    print("\nSummary Statistics:")
    print(f"Number of tasks: {num_tasks}")
    print(f"Number of models: {num_models}")
    
    if 'test_accuracy' in summary_df.columns:
        best_model_idx = summary_df['test_accuracy'].idxmax()
        best_model = summary_df.iloc[best_model_idx]
        print(f"\nBest overall model: {best_model['model_name']} on {best_model['task_name']}")
        print(f"Test accuracy: {best_model['test_accuracy']:.4f}")
        
        if 'test_f1_score' in best_model:
            print(f"Test F1 score: {best_model['test_f1_score']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 