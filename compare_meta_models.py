import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_results(results_dir, model_types, test_types):
    """
    Load results from JSON files for different models and test types
    
    Args:
        results_dir: Root directory containing results
        model_types: List of model types to compare
        test_types: List of test types to compare
        
    Returns:
        DataFrame with results
    """
    results_data = []
    
    for model_type in model_types:
        for test_type in test_types:
            results_file = os.path.join(results_dir, f"{model_type}_MotionSourceRecognition", f"{test_type}_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract key metrics
                accuracy = results.get('accuracy', 0)
                std_accuracy = results.get('std_accuracy', 0)
                loss = results.get('loss', 0)
                n_tasks = results.get('n_tasks', 0)
                adaptation_curve = results.get('adaptation_curve', [])
                
                # Create nice test type name
                test_name = test_type.replace('_', ' ').title()
                if "Adapt" in test_name:
                    if "1shot" in test_type:
                        test_name = test_name.replace("1Shot", "1-Shot")
                    elif "5shot" in test_type:
                        test_name = test_name.replace("5Shot", "5-Shot")
                
                # Store in results array
                results_data.append({
                    'Model': model_type.upper(),
                    'Test Type': test_name,
                    'Accuracy': accuracy,
                    'Std Accuracy': std_accuracy,
                    'Loss': loss,
                    'Tasks': n_tasks,
                    'Adaptation Curve': adaptation_curve
                })
    
    # Convert to DataFrame
    if results_data:
        return pd.DataFrame(results_data)
    else:
        return pd.DataFrame(columns=['Model', 'Test Type', 'Accuracy', 'Std Accuracy', 'Loss', 'Tasks', 'Adaptation Curve'])

def plot_accuracy_comparison(df, output_dir):
    """Plot accuracy comparison across models and test types"""
    if df.empty:
        print("No data to plot")
        return
    
    # Get unique test types and models
    test_types = df['Test Type'].unique()
    models = df['Model'].unique()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Compute positions for grouped bars
    bar_width = 0.8 / len(models)
    positions = np.arange(len(test_types))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        
        # Ensure data is in the same order as test_types
        accuracies = []
        std_accuracies = []
        
        for test_type in test_types:
            test_data = model_data[model_data['Test Type'] == test_type]
            if not test_data.empty:
                accuracies.append(test_data.iloc[0]['Accuracy'])
                std_accuracies.append(test_data.iloc[0]['Std Accuracy'])
            else:
                accuracies.append(0)
                std_accuracies.append(0)
        
        # Plot bars with error bars
        plt.bar(
            positions + i * bar_width - (len(models) - 1) * bar_width / 2,
            accuracies,
            bar_width,
            label=model,
            yerr=std_accuracies,
            capsize=4
        )
    
    # Set labels and title
    plt.xlabel('Test Type')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison Across Test Types')
    plt.xticks(positions, test_types, rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def plot_adaptation_curves(df, output_dir):
    """Plot adaptation curves for each model and test type"""
    if df.empty or 'Adaptation Curve' not in df.columns:
        print("No adaptation curve data to plot")
        return
    
    # Get unique models
    models = df['Model'].unique()
    
    # Group by test type
    test_types = df['Test Type'].unique()
    
    for test_type in test_types:
        test_data = df[df['Test Type'] == test_type]
        
        if test_data.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        
        for model in models:
            model_data = test_data[test_data['Model'] == model]
            
            if not model_data.empty and len(model_data.iloc[0]['Adaptation Curve']) > 0:
                adaptation_curve = model_data.iloc[0]['Adaptation Curve']
                plt.plot(range(1, len(adaptation_curve) + 1), adaptation_curve, marker='o', label=model)
        
        # Set labels and title
        plt.xlabel('Adaptation Steps')
        plt.ylabel('Query Set Accuracy')
        plt.title(f'Adaptation Curves for {test_type}')
        plt.grid(True)
        plt.legend(loc='best')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'adaptation_curve_{test_type.replace(" ", "_").lower()}.png'), dpi=300)
        plt.close()

def create_summary_table(df, output_dir):
    """Create summary table of results"""
    if df.empty:
        print("No data for summary table")
        return
    
    # Create pivot table
    pivot = df.pivot_table(
        values='Accuracy',
        index='Test Type',
        columns='Model',
        aggfunc='first'
    )
    
    # Save as CSV
    pivot.to_csv(os.path.join(output_dir, 'accuracy_summary.csv'))
    
    # Create formatted table for visualization
    plt.figure(figsize=(12, len(pivot) * 0.6))
    plt.axis('off')
    
    # Create table
    table = plt.table(
        cellText=np.around(pivot.values, 4),
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2] * len(pivot.columns)
    )
    
    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Accuracy Summary by Model and Test Type', fontsize=16, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create stderr table for visualization
    stderr_pivot = df.pivot_table(
        values='Std Accuracy',
        index='Test Type',
        columns='Model',
        aggfunc='first'
    )
    
    # Save as CSV
    stderr_pivot.to_csv(os.path.join(output_dir, 'std_accuracy_summary.csv'))

def main():
    parser = argparse.ArgumentParser(description='Compare meta-learning model performance')
    parser.add_argument('--results_dir', type=str, default='results/meta_testing',
                        help='Directory containing test results')
    parser.add_argument('--model_types', type=str, nargs='+', 
                        default=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                        help='Model types to compare')
    parser.add_argument('--test_types', type=str, nargs='+',
                        default=['test', 'cross_env', 'cross_user', 'cross_device',
                                 'adapt_1shot', 'adapt_5shot'],
                        help='Test types to compare')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='Directory to save comparison results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results_df = load_results(args.results_dir, args.model_types, args.test_types)
    
    if results_df.empty:
        print("No results found. Make sure you've run test_meta_models.py for the specified models and test types.")
        return
    
    # Print available models and test types
    available_models = results_df['Model'].unique()
    available_tests = results_df['Test Type'].unique()
    
    print(f"Found results for {len(available_models)} models: {', '.join(available_models)}")
    print(f"Found results for {len(available_tests)} test types: {', '.join(available_tests)}")
    
    # Plot accuracy comparison
    print("Plotting accuracy comparison...")
    plot_accuracy_comparison(results_df, output_dir)
    
    # Plot adaptation curves
    print("Plotting adaptation curves...")
    plot_adaptation_curves(results_df, output_dir)
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(results_df, output_dir)
    
    print(f"Comparison results saved to {output_dir}")

if __name__ == "__main__":
    main() 