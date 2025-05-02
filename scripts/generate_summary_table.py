#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Summary Table Script

This script generates a summary table of performance metrics across different models and tasks.
It collects results from individual model summary files and creates consolidated CSV tables.

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
    return parser.parse_args()

def find_summary_files(results_dir):
    """
    Find all summary files in the results directory structure.
    
    Args:
        results_dir: Base directory containing results
        
    Returns:
        List of paths to summary files
    """
    # Pattern matches any summary.json file in the directory structure
    pattern = os.path.join(results_dir, "**", "*_summary.json")
    return glob(pattern, recursive=True)

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
        return data
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
    key_metrics = ['model_name', 'task_name', 'best_val_accuracy']
    test_acc_cols = [col for col in df.columns if 'test' in col and 'accuracy' in col]
    test_f1_cols = [col for col in df.columns if 'test' in col and 'f1_score' in col]
    
    compact_cols = key_metrics + test_acc_cols + test_f1_cols
    if all(col in df.columns for col in key_metrics) and len(compact_cols) > 3:
        tables['compact'] = df[compact_cols]
    
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
                aggfunc=np.mean  # In case there are duplicates
            )
            
            # Add a row with the average across tasks
            pivot.loc['Average'] = pivot.mean()
            
            # Add a column with the average across models
            pivot['Average'] = pivot.mean(axis=1)
            
            pivot_tables[f"pivot_{metric}"] = pivot
        except Exception as e:
            print(f"Could not create pivot table for {metric}: {e}")
    
    return pivot_tables

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
    """Main function"""
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    # Find all summary files
    print(f"Searching for summary files in {args.results_dir}...")
    summary_files = find_summary_files(args.results_dir)
    print(f"Found {len(summary_files)} summary files")
    
    if not summary_files:
        print("No summary files found. Exiting.")
        return 1
    
    # Create summary table
    print("Creating summary table...")
    df = create_summary_table(summary_files)
    
    if df is None or df.empty:
        print("Could not create summary table. Exiting.")
        return 1
    
    # Generate specialized tables
    print("Generating specialized tables...")
    tables = generate_specialized_tables(df)
    
    # Create pivot tables
    print("Creating pivot tables...")
    pivot_tables = create_pivot_tables(df)
    
    # Combine all tables
    all_tables = {**tables, **pivot_tables}
    
    # Save tables
    print(f"Saving tables to {output_dir}...")
    save_tables(all_tables, output_dir)
    
    print("Summary tables generated successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 