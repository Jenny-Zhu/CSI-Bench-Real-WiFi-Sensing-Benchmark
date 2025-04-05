import torch
import torch.nn as nn
import torch.optim as optim
import higher
import argparse
from MAMLTaskDataset import CSITaskDataset
from Networks import CSI2DCNN
from util import evaluate

def maml_train(model, task_dataset, steps=1000, tasks_per_batch=4, inner_lr=0.01, meta_lr=0.001):
    """
    Meta-learning training loop using Model-Agnostic Meta-Learning (MAML).

    :param model: The neural network model (e.g., CSI2DCNN)
    :param task_dataset: The task sampler that returns few-shot tasks (support + query sets)
    :param steps: Number of meta-training steps
    :param tasks_per_batch: Number of tasks to train on in each meta step
    :param inner_lr: Learning rate for inner-loop (task-specific adaptation)
    :param meta_lr: Learning rate for outer-loop (meta-update)
    """
    # Outer-loop optimizer
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)

    # Loss function for classification
    loss_fn = nn.CrossEntropyLoss()

    for meta_step in range(steps):
        meta_opt.zero_grad()
        meta_loss = 0

        # Iterate over multiple tasks in each meta-step
        for _ in range(tasks_per_batch):
            # Get support and query sets for one few-shot classification task
            x_s, y_s, x_q, y_q = task_dataset.sample_task()
            x_s, x_q = x_s.float(), x_q.float()

            # Inner-loop optimizer (per task)
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            # Use 'higher' to enable differentiable inner-loop updates
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # === Inner Loop: adapt the model using support set ===
                logits_s = fmodel(x_s)
                loss_s = loss_fn(logits_s, y_s)
                diffopt.step(loss_s)  # Perform one or more gradient steps

                # === Outer Loop: evaluate on query set ===
                logits_q = fmodel(x_q)
                loss_q = loss_fn(logits_q, y_q)
                meta_loss += loss_q

        # === Meta Update ===
        meta_loss.backward()  # Backprop through inner updates
        meta_opt.step()       # Update meta-parameters

        # Print progress every 100 steps
        if meta_step % 100 == 0:
            acc = evaluate(model, x_q, y_q)
            print(f"[Step {meta_step}] Meta Loss: {meta_loss.item():.4f} | Query Acc: {acc:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Choose the task: good/bad or motion/empty classification
    parser.add_argument('--task', type=str, choices=['goodbad', 'motionempty'], default='goodbad',
                        help="Task type: 'goodbad' or 'motionempty'")
    parser.add_argument('--folder', type=str, required=True,
                        help="Path to folder containing .mat CSI files")

    args = parser.parse_args()

    # Few-shot task config  #
    k_shot = 5           # Number of support samples per class
    q_query = 15         # Number of query samples per class
    resize_height = 64   # Standardized height (e.g., # of subcarriers)

    # Define label keywords per task #
    if args.task == 'goodbad':
        label_keywords = {'good': 0, 'bad': 1}
    elif args.task == 'motionempty':
        label_keywords = {'empty': 0, 'motion': 1}
    else:
        raise ValueError("Invalid task name. Choose from: goodbad, motionempty")

    # Initialize dataset & model #
    task_dataset = CSITaskDataset(
        folder_path=args.folder,
        k_shot=k_shot,
        q_query=q_query,
        resize_height=resize_height,
        label_keywords=label_keywords
    )

    # Infer shape (H, W) from one support sample
    x_s, _, _, _ = task_dataset.run_task()
    _, _, H, W = x_s.shape

    # Initialize 2D CNN model for CSI classification
    model = CSI2DCNN(input_size=(H, W))

    # -------------------- #
    # Start MAML training  #
    # -------------------- #
    maml_train(model, task_dataset)
