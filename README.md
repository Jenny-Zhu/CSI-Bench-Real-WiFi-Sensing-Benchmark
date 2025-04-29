# WiFi Sensing Analysis Tools

This repository contains a set of tools for analyzing WiFi sensing model performance results directly from Amazon S3, eliminating the need to download large result files first.

## Overview

These tools allow you to:

1. **Directly analyze results from S3** without downloading files first
2. **Generate comparison tables** across models and tasks
3. **Create visualizations** of model performance 
4. **Plot training/validation loss curves** to track model convergence
5. **Generate comprehensive reports** with insights and recommendations

## Main Analysis Tool

### S3 WiFi Analyzer (`s3_wifi_analyzer.py`)

This comprehensive tool handles all aspects of WiFi sensing experiment analysis directly from S3.

```bash
python s3_wifi_analyzer.py --s3-dir s3://bucket/results/ --tasks FourClass HumanNonhuman --models Transformer ResNet18
```

#### Features

- **Direct S3 Access**: Reads results directly from S3, no need to download files first
- **Summary Tables**: Creates comprehensive tables of model performance across tasks
- **Learning Curves**: Generates training and validation loss/accuracy curves
- **Model Comparisons**: Visualizes performance comparisons between models across tasks
- **Performance Metrics**: Analyzes accuracy, precision, recall, F1 score, and training time
- **Aggregation**: Combines results from multiple runs using mean, median, or max

#### Common Options

```bash
# Analyze specific tasks and models
python s3_wifi_analyzer.py --s3-dir s3://bucket/results/ --tasks FourClass HumanNonhuman --models Transformer ResNet18

# Specify output directory
python s3_wifi_analyzer.py --s3-dir s3://bucket/results/ --output-dir ./my_analysis/

# Disable curve plotting to speed up analysis
python s3_wifi_analyzer.py --s3-dir s3://bucket/results/ --no-plot-curves

# Change aggregation method for multiple runs
python s3_wifi_analyzer.py --s3-dir s3://bucket/results/ --agg-method max
```

#### Full Parameter List

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--s3-dir` | S3 URI containing results (Required) | None |
| `--output-dir` | Local directory for analysis outputs | ./analysis_results |
| `--tasks` | Specific tasks to analyze | All tasks |
| `--models` | Specific models to analyze | All models |
| `--metrics` | Performance metrics to analyze | accuracy, precision, recall, f1, training_time |
| `--plot-curves` | Plot training/validation loss curves | True |
| `--plot-comparisons` | Plot model performance comparisons | True |
| `--agg-method` | Method to aggregate multiple runs (mean, median, max) | mean |

## Result Summarization Process

The current result summarization workflow efficiently analyzes WiFi sensing experiment results:

1. **Data Discovery**: The analyzer automatically discovers experiments in the specified S3 directory by navigating the hierarchical structure of task/model/job directories.

2. **File Extraction**: For each experiment, the tool locates and extracts data from:
   - `classification_report.csv` - Contains model performance metrics
   - `training_results.csv` - Contains training history data
   - `hyperparams.json` - Contains experiment configuration parameters

3. **Metrics Compilation**: Performance metrics are extracted and compiled into comprehensive tables:
   - Overall summary table for all tasks and models
   - Per-task performance tables
   - Per-model average performance across tasks

4. **Visualization Generation**:
   - Training/validation loss curves for each experiment
   - Accuracy curves for each experiment
   - Model comparison heatmaps across tasks
   - Bar charts of model performance

5. **Report Generation**: A Markdown summary report is created with key findings and links to visualizations.

## Expected S3 Directory Structure

The tool expects results to be organized in the following hierarchy:

```
s3://bucket/results/
  ├── Task1/
  │    ├── Model1/
  │    │    ├── Job1/
  │    │    │    ├── classification_report.csv
  │    │    │    └── training_results.csv
  │    │    └── Job2/
  │    └── Model2/
  └── Task2/
       ├── Model1/
       └── Model2/
```

## Requirements

- Python 3.7+
- boto3
- pandas
- matplotlib
- seaborn
- numpy

## Installation

```bash
pip install boto3 pandas matplotlib seaborn numpy
```

## AWS Credentials

The tool requires AWS credentials with S3 access. Configure your credentials using one of these methods:

1. **AWS CLI**: Run `aws configure`
2. **Environment variables**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
3. **Credentials file**: Create a file at `~/.aws/credentials`

## Troubleshooting

1. **No results found**: Check the S3 path and verify the directory structure
2. **Access denied**: Verify AWS credentials and permissions
3. **Empty results**: Ensure the classification_report.csv and training_results.csv files exist
4. **Memory errors**: For large datasets, analyze fewer tasks/models at once
