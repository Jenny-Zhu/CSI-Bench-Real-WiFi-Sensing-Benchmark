import torch
import torch.nn as nn
import torch.optim as optim
import higher
import matplotlib.pyplot as plt 
import torch.nn.functional as F

def evaluate(fmodel, x, y):
    logits = fmodel(x)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean().item()
    return acc

def preprocess_csi(raw_csi):
    # Placeholder for real preprocessing
    return raw_csi

class LSTMOptimizer(nn.Module):
    def __init__(self, hidden_size=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(2, hidden_size)  # input: (grad, param)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, grad, param, hx, cx):
        """
        grad, param: flattened tensor (num_params,)
        hx, cx: hidden and cell states (num_params, hidden_size)
        """
        input_combined = torch.stack([grad, param], dim=1)  # shape (num_params, 2)
        hx, cx = self.lstm(input_combined, (hx, cx))  # LSTM update
        delta = self.fc(hx).squeeze(1)  # Predict parameter update
        return delta, hx, cx

def model_forward_with_weights(model, x, fast_weights):
    """
    Apply new fast_weights manually during forward pass.
    """
    x = F.relu(F.max_pool2d(F.conv2d(x, fast_weights['conv1.weight'], fast_weights['conv1.bias'], padding=1), 2))
    x = F.relu(F.max_pool2d(F.conv2d(x, fast_weights['conv2.weight'], fast_weights['conv2.bias'], padding=1), 2))

    x = x.view(x.size(0), -1)
    x = F.relu(F.linear(x, fast_weights['fc1.weight'], fast_weights['fc1.bias']))
    logits = F.linear(x, fast_weights['fc2.weight'], fast_weights['fc2.bias'])

    return logits



def maml_train(model, task_dataset, device, steps=1000, tasks_per_batch=4, inner_lr=0.01, meta_lr=0.001):
    
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr) # Create the outer-loop (meta) optimizer
    
    loss_fn = nn.CrossEntropyLoss() # Define loss function (CrossEntropy for classification tasks)

    meta_losses = []

    # Start meta-training loop
    for meta_step in range(steps):
        meta_opt.zero_grad() # Clear previous outer-loop gradients
        meta_loss = 0 # Initialize meta-loss for this meta-step

        # For each meta-step, train on multiple sampled tasks (task batch)
        for _ in range(tasks_per_batch):
            x_s, y_s, x_q, y_q = task_dataset.sample_task() # Sample one few-shot task (support set + query set)
            x_s, y_s, x_q, y_q = x_s.to(device), y_s.to(device), x_q.to(device), y_q.to(device)
           
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)  # Create inner-loop optimizer (SGD) for fast adaptation on this task
            
            # Enter "higher" context to make inner-loop optimization differentiable
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # ----- Inner Loop (Adaptation Step) -----
                logits_s = fmodel(x_s) # Forward pass on support set

                loss_s = loss_fn(logits_s, y_s) # Compute loss on support set

                diffopt.step(loss_s) # Take one or more gradient steps using support loss

                # ----- Evaluate Adapted Model on Query Set -----
                logits_q = fmodel(x_q) # Forward pass on query set
                loss_q = loss_fn(logits_q, y_q) # Loss on query set
                meta_loss += loss_q # Accumulate query losses across tasks

        # ----- Outer Loop (Meta Update) -----
        meta_loss.backward() # Backpropagate through all inner loops (computes meta-gradients)
        meta_opt.step() # Update meta-parameters using accumulated meta-gradients

        meta_losses.append(meta_loss.item())


        # if meta_step % 100 == 0:
            # Evaluate the current model (without adaptation) on query set
        acc = evaluate(model, x_q, y_q)
        print(f"[Step {meta_step}] Meta Loss: {meta_loss.item():.4f} | Query Acc: {acc:.2f}")

    # New: After training, plot loss
    plt.plot(meta_losses)
    plt.xlabel('Meta Step')
    plt.ylabel('Meta Loss')
    plt.title('Meta-Loss over Training')
    plt.grid(True)
    plt.show()

def lstm_meta_train(model, meta_optimizer, task_dataset, device, steps=1000, tasks_per_batch=4):
    """
    LSTM Meta-Learner training loop.
    """
    loss_fn = nn.CrossEntropyLoss()
    meta_optim = optim.Adam(meta_optimizer.parameters(), lr=1e-3)  # optimizer for meta-optimizer

    for meta_step in range(steps):
        meta_optim.zero_grad()
        meta_loss = 0

        for _ in range(tasks_per_batch):
            # Sample a few-shot task
            x_s, y_s, x_q, y_q = task_dataset.sample_task()
            x_s, y_s, x_q, y_q = x_s.to(device), y_s.to(device), x_q.to(device), y_q.to(device)

            # Copy current model parameters
            fast_weights = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}

            # Init hidden and cell states for LSTM
            num_params = sum(p.numel() for p in fast_weights.values())
            hx = torch.zeros(num_params, meta_optimizer.hidden_size).to(device)
            cx = torch.zeros(num_params, meta_optimizer.hidden_size).to(device)

            # Compute loss on support set
            logits_s = model_forward_with_weights(model, x_s, fast_weights)
            loss_s = loss_fn(logits_s, y_s)
            grads = torch.autograd.grad(loss_s, fast_weights.values(), create_graph=True)

            # Flatten parameters and gradients
            flat_params = torch.cat([p.view(-1) for p in fast_weights.values()])
            flat_grads = torch.cat([g.view(-1) for g in grads])

            # LSTM meta-optimizer predicts updates
            delta, hx, cx = meta_optimizer(flat_grads, flat_params, hx, cx)
            updated_flat_params = flat_params + delta

            # Rebuild fast_weights dictionary
            new_fast_weights = {}
            pointer = 0
            for (name, param) in fast_weights.items():
                numel = param.numel()
                new_fast_weights[name] = updated_flat_params[pointer:pointer+numel].view_as(param)
                pointer += numel

            # Compute query loss with updated parameters
            logits_q = model_forward_with_weights(model, x_q, new_fast_weights)
            loss_q = loss_fn(logits_q, y_q)
            meta_loss += loss_q

        # Meta-optimizer update
        meta_loss.backward()
        meta_optim.step()

        if meta_step % 100 == 0:
            print(f"[Step {meta_step}] Meta Loss: {meta_loss.item():.4f}")