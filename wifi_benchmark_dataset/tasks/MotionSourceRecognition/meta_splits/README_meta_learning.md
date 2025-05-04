# Meta-Learning Splits for CSI Motion Source Recognition

This repository contains data splits for both supervised learning and meta-learning approaches to motion source recognition using CSI data. The meta-learning splits are designed to evaluate how well models can adapt to new users, environments, or activities with limited data.

## Data Structure

The data is organized as follows:

```
motion_source_recognition/
├── metadata/
│   └── sample_metadata.csv       # Comprehensive metadata for all samples
├── splits/                       # Supervised learning splits
│   ├── train_id.json             # Training samples
│   ├── val_id.json               # Validation samples
│   ├── test_id.json              # In-distribution test samples
│   ├── test_cross_env.json       # Cross-environment test samples
│   ├── test_cross_user.json      # Cross-user test samples
│   ├── test_cross_device.json    # Cross-device test samples (if available)
│   └── ...                       # Other specialized test sets
└── meta_splits/                  # Meta-learning splits
    ├── train_tasks.json          # Meta-learning training tasks
    ├── val_tasks.json            # Meta-learning validation tasks
    ├── test_tasks.json           # Meta-learning test tasks (in-distribution)
    ├── test_cross_env_tasks.json # Meta-learning cross-environment test tasks
    ├── test_cross_user_tasks.json # Meta-learning cross-user test tasks
    ├── test_cross_device_tasks.json # Meta-learning cross-device test tasks (if available)
    ├── adapt_1shot_tasks.json    # 1-shot adaptation tasks for unseen users
    ├── adapt_5shot_tasks.json    # 5-shot adaptation tasks for unseen users
    ├── cross_env_adapt_1shot_tasks.json  # 1-shot adaptation for cross-environment
    ├── cross_env_adapt_5shot_tasks.json  # 5-shot adaptation for cross-environment
    ├── cross_user_adapt_1shot_tasks.json # 1-shot adaptation for cross-user
    ├── cross_user_adapt_5shot_tasks.json # 5-shot adaptation for cross-user
    ├── cross_device_adapt_1shot_tasks.json # 1-shot adaptation for cross-device (if available)
    └── cross_device_adapt_5shot_tasks.json # 5-shot adaptation for cross-device (if available)
```

## Meta-Learning Data Format

Each meta-learning file contains a list of tasks. Each task includes:

```json
{
  "task_id": "task_Human_1",    // Unique task identifier
  "subject": "Human",           // Subject label 
  "user": "user_123",           // User identifier
  "support": [                  // Support set sample IDs (for training)
    "Human_user_123_...",
    ...
  ],
  "query": [                    // Query set sample IDs (for evaluation)
    "Human_user_123_...",
    ...
  ]
}
```

## How to Use

### 1. Supervised Learning Approach

Use the splits in `motion_source_recognition/splits/` for a traditional supervised learning approach:
- Train on `train_id.json`
- Validate on `val_id.json`
- Test on `test_id.json` (in-distribution)
- Evaluate generalization to unseen environments with `test_cross_env.json`
- Evaluate generalization to unseen users with `test_cross_user.json`
- Evaluate generalization to unseen devices with `test_cross_device.json` (if available)

### 2. Meta-Learning Approach

Meta-learning trains on episodes/tasks rather than individual samples:

1. **Train a meta-learning model**:
   - Use `train_tasks.json` for training
   - Each task contains a support set (for task adaptation) and a query set (for evaluation)
   - The model learns to quickly adapt to new tasks with limited data

2. **Validate the meta-learning model**:
   - Use `val_tasks.json` for validation

3. **Test the meta-learning model**:
   - In-distribution testing: `test_tasks.json`
   - Cross-environment testing: `test_cross_env_tasks.json`
   - Cross-user testing: `test_cross_user_tasks.json`
   - Cross-device testing: `test_cross_device_tasks.json` (if available)
   - Few-shot adaptation testing: `adapt_1shot_tasks.json` and `adapt_5shot_tasks.json`
   - Cross-environment adaptation: `cross_env_adapt_1shot_tasks.json` and `cross_env_adapt_5shot_tasks.json`
   - Cross-user adaptation: `cross_user_adapt_1shot_tasks.json` and `cross_user_adapt_5shot_tasks.json`
   - Cross-device adaptation: `cross_device_adapt_1shot_tasks.json` and `cross_device_adapt_5shot_tasks.json` (if available)

### 3. Comparing Approaches

To compare supervised learning with meta-learning:

1. First evaluate your supervised model on the test sets
2. Then evaluate your meta-learning model on the corresponding meta-learning test tasks
3. Compare performance across different scenarios, especially:
   - In-distribution test performance
   - Cross-environment generalization
   - Cross-user generalization
   - Cross-device generalization (if available)
   - Adaptation to new users with limited data (1-shot and 5-shot)

## Implementation Example

Here's a simplified example of how to implement meta-learning with these splits:

```python
# Pseudocode for meta-learning training
for epoch in range(epochs):
    for task in train_tasks:
        # Get support and query sets for this task
        support_samples = load_samples(task["support"])
        query_samples = load_samples(task["query"])
        
        # Meta-learning inner loop: adapt to the support set
        adapted_model = model.adapt(support_samples)
        
        # Meta-learning outer loop: evaluate on query set and update meta-parameters
        loss = adapted_model.evaluate(query_samples)
        optimizer.step(loss)
```

## Generating the Splits

The meta-learning splits were generated using the `meta_learning_splits_generator.py` script, which:
1. Loads the comprehensive metadata and existing supervised learning splits
2. Creates episodic tasks for meta-learning training, validation, and testing
3. Creates specialized adaptation tasks for few-shot learning evaluation
4. Saves all tasks in JSON format in the `meta_splits` directory

## Evaluation Scenarios

This dataset supports several challenging evaluation scenarios:

1. **Cross-User Evaluation**: Testing on users not seen during training
2. **Cross-Environment Evaluation**: Testing in environments not seen during training
3. **Cross-Device Evaluation**: Testing on devices not seen during training (if available)
4. **Few-Shot Adaptation**: Testing with limited support samples (1 or 5 shots)

These scenarios allow for a comprehensive comparison between traditional supervised learning and meta-learning approaches, particularly in terms of generalization and adaptation capabilities.

## References

For more information on meta-learning approaches:
- Model-Agnostic Meta-Learning (MAML): [Finn et al., 2017](https://arxiv.org/abs/1703.03400)
- Prototypical Networks: [Snell et al., 2017](https://arxiv.org/abs/1703.05175)
- Reptile: [Nichol et al., 2018](https://arxiv.org/abs/1803.02999) 