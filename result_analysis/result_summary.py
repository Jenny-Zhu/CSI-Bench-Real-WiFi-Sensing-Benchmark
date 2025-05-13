import os
import json
import pandas as pd
import glob
from typing import List, Dict, Any


def create_result_summary(
    data_dir: str,
    pipelines: List[str],
    tasks: List[str],
    models: List[str],
    output_csv: str = "results_summary.csv"
) -> pd.DataFrame:
    """
    Create a summary of the training results in a pandas DataFrame.
    
    Args:
        data_dir: Base directory containing the results
        pipelines: List of pipeline names to include (e.g., ["supervised", "self_supervised"])
        tasks: List of task names to include (e.g., ["ProximityRecognition", "RoomRecognition"])
        models: List of model names to include (e.g., ["lstm", "mlp", "cnn"])
        output_csv: Path to save the CSV output (default: "results_summary.csv")
        
    Returns:
        pd.DataFrame: DataFrame containing the summarized results
    """
    results = []
    
    for pipeline in pipelines:
        for task in tasks:
            for model in models:
                model_path = os.path.join(data_dir, pipeline, task, model)
                
                if not os.path.exists(model_path):
                    print(f"Path does not exist: {model_path}")
                    continue
                
                best_perf_path = os.path.join(model_path, "best_performance.json")
                
                if not os.path.exists(best_perf_path):
                    print(f"Best performance file not found: {best_perf_path}")
                    continue
                
                try:
                    with open(best_perf_path, 'r') as f:
                        best_perf = json.load(f)
                    
                    result_row = {
                        'pipeline': pipeline,
                        'task': task,
                        'model': model
                    }
                    
                    # Add experiment_id, best_epoch, and avg_test_accuracy
                    for field in ['experiment_id', 'best_epoch', 'avg_test_accuracy']:
                        if field in best_perf:
                            result_row[field] = best_perf[field]
                    
                    # Extract learning rate and weight decay from config file
                    if 'experiment_id' in best_perf:
                        experiment_id = best_perf['experiment_id']
                        experiment_path = os.path.join(model_path, f"{experiment_id}")
                        config_file = os.path.join(experiment_path, f"{model}_{task}_config.json")
                        
                        if os.path.exists(config_file):
                            try:
                                with open(config_file, 'r') as f:
                                    config = json.load(f)
                                
                                # Extract learning rate and weight decay
                                if 'learning_rate' in config:
                                    result_row['learning_rate'] = config['learning_rate']
                                    print(f"Learning rate: {result_row['learning_rate']}")
                                if 'weight_decay' in config:
                                    result_row['weight_decay'] = config['weight_decay']
                                    print(f"Weight decay: {result_row['weight_decay']}")
                            except Exception as e:
                                print(f"Error reading config file {config_file}: {e}")
                        else:
                            print(f"Config file not found: {config_file}")
                    
                    # Extract test metrics for each test dataset
                    if 'test_metrics' in best_perf:
                        for test_name, metrics in best_perf['test_metrics'].items():
                            for metric_name, metric_value in metrics.items():
                                col_name = f"{test_name}_{metric_name}"
                                result_row[col_name] = metric_value
                    
                    results.append(result_row)
                    
                except Exception as e:
                    print(f"Error processing {best_perf_path}: {e}")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV if there are results
    if not df.empty and output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return df


if __name__ == "__main__":
    # Example usage - you can modify these values
    data_dir = "C:\\Users\\weiha\\Desktop\\benchmark_result"
    
    # Pipelines to analyze
    pipelines = ["supervised"]
    
    # Tasks to analyze
    tasks = ["ProximityRecognition", "HumanIdentification", "HumanActivityRecognition",'MotionSourceRecognition','FallDetection']
    # tasks = ["ProximityRecognition", "HumanIdentification"]
    # Models to analyze
    models = ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"]
    
    # Create the summary
    result_df = create_result_summary(
        data_dir=data_dir,
        pipelines=pipelines,
        tasks=tasks,
        models=models,
        output_csv="results_summary.csv"
    )
    
    # Display the results
    if not result_df.empty:
        print("\nResults Summary:")
        print(result_df)
    else:
        print("No results found matching the specified criteria.") 