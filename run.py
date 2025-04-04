import torch
import torch.nn as nn
import torch.optim as optim
import higher
from MAMLTaskDataset import DummyCSITaskDataset
from Networks import CSINet
from util import evaluate

def maml_train(model, task_sampler, steps=1000, tasks_per_batch=4, inner_lr=0.01, meta_lr=0.001):
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss()

    for meta_step in range(steps):
        meta_opt.zero_grad()
        meta_loss = 0

        for _ in range(tasks_per_batch):
            x_s, y_s, x_q, y_q = task_sampler.sample_task()
            x_s, x_q = x_s.float(), x_q.float()

            with higher.innerloop_ctx(model, meta_opt, copy_initial_weights=False) as (fmodel, diffopt):
                logits_s = fmodel(x_s)
                loss_s = loss_fn(logits_s, y_s)
                diffopt.step(loss_s)

                logits_q = fmodel(x_q)
                loss_q = loss_fn(logits_q, y_q)
                meta_loss += loss_q

        meta_loss.backward()
        meta_opt.step()

        if meta_step % 100 == 0:
            acc = evaluate(model, x_q, y_q)
            print(f"Step {meta_step}: Meta Loss = {meta_loss.item():.4f}, Query Acc = {acc:.4f}")

if __name__ == '__main__':
    input_len = 100
    model = CSINet(input_len=input_len)
    task_sampler = DummyCSITaskDataset(input_len=input_len)
    maml_train(model, task_sampler)
