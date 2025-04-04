import torch
import torch.nn as nn
import torch.optim as optim
import higher
from MAMLTaskDataset import CSITaskDataset
from Networks import CSI2DCNN
from util import evaluate

def maml_train(model, task_dataset, steps=1000, tasks_per_batch=4, inner_lr=0.01, meta_lr=0.001):
    """
    MAML training loop for few-shot learning on CSI data.

    :param model: The neural network model 
    :param task_dataset: CSITaskDataset object that provides few-shot tasks
    :param steps: Number of meta-training steps (outer loop iterations)
    :param tasks_per_batch: Number of tasks to sample per meta-step
    :param inner_lr: Learning rate used for inner-loop adaptation (support set)
    :param meta_lr: Learning rate used for outer-loop meta-update (query set)
    """
    # Optimizer for meta-level parameters (outer loop)
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)

    # Classification loss
    loss_fn = nn.CrossEntropyLoss()

    for meta_step in range(steps):
        meta_opt.zero_grad()  # Reset outer-loop gradients
        meta_loss = 0         # Accumulate loss from each task

        # Sample and train on multiple tasks per meta-step
        for _ in range(tasks_per_batch):
            # Sample a few-shot task (support and query sets)
            x_s, y_s, x_q, y_q = task_dataset.sample_task()
            x_s, x_q = x_s.float(), x_q.float()

            # Inner-loop optimizer (fresh for each task)
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            # Use 'higher' to create a differentiable inner-loop
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # === Inner Loop: Adapt to support set ===
                logits_s = fmodel(x_s)                  # Forward pass on support set
                loss_s = loss_fn(logits_s, y_s)         # Compute loss on support set
                diffopt.step(loss_s)                    # Perform one gradient step

                # === Outer Loop: Evaluate on query set ===
                logits_q = fmodel(x_q)                  # Forward pass on query set
                loss_q = loss_fn(logits_q, y_q)         # Compute loss on query set
                meta_loss += loss_q                     # Accumulate for outer update

        # === Meta Update ===
        meta_loss.backward()  # Backprop through inner updates
        meta_opt.step()       # Update meta-parameters

        # Optional: Print progress every 100 steps
        if meta_step % 100 == 0:
            acc = evaluate(model, x_q, y_q)  # Evaluate on the latest task’s query set
            print(f"[Step {meta_step}] Meta Loss: {meta_loss.item():.4f} | Query Acc: {acc:.2f}")


if __name__ == '__main__':
    folder_path = '/data_folder'
    k_shot = 5
    q_query = 15
    resize_height = 64

    # Initialize task dataset
    task_dataset = CSITaskDataset(folder_path=folder_path, k_shot=k_shot, q_query=q_query, resize_height=resize_height)

    # Infer input shape from one batch
    x_s, _, _, _ = task_dataset.sample_task()
    _, _, H, W = x_s.shape

    # Init model with inferred shape
    from Networks import CSI2DCNN
    model = CSI2DCNN(input_size=(H, W))

    # Run MAML training
    maml_train(model, task_dataset)