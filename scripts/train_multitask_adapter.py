import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from load.supervised.benchmark_loader import load_benchmark_supervised
from model.multitask.models import MultiTaskAdapterModel
from scripts.train_supervised import MODEL_TYPES
from engine.supervised.task_trainer import TaskTrainer



def parse_args():
    parser = argparse.ArgumentParser("Multi-task Adapter Training", add_help=False)
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of task names')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Backbone model type, same as supervised pipeline')
    parser.add_argument('--data_dir', '--training_dir', dest='data_dir', type=str,
                        default='wifi_benchmark_dataset',
                        help='Dataset root directory (alias: --training_dir)')
    parser.add_argument('--win_len', type=int, help='window length for MLP/VIT')
    parser.add_argument('--feature_size', type=int, help='feature size for LSTM/Transformer/VIT')
    parser.add_argument('--in_channels', type=int, help='input channels for ResNet18')
    parser.add_argument('--emb_dim', type=int, help='embedding dim for VIT/Transformer')
    parser.add_argument('--dropout', type=float, help='dropout for VIT/Transformer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='checkpoints/multitask')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience (in epochs) for early stopping')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare loaders
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    task_classes = {}
    for task in args.tasks.split(','):
        data = load_benchmark_supervised(
            dataset_root=args.data_dir,
            task_name=task,
            batch_size=args.batch_size
        )
        ld = data['loaders']
        train_loaders[task] = ld['train']
        val_loaders[task] = ld.get('val', ld['train'])
        test_loaders[task] = {k: v for k, v in ld.items() if k.startswith('test')}
        task_classes[task] = data['num_classes']

    # Infer feature_size if needed
    first_loader = next(iter(train_loaders.values()))
    sample_x, _ = next(iter(first_loader))
    if args.feature_size is None:
        args.feature_size = sample_x.shape[-1]

    # Build backbone
    ModelClass = MODEL_TYPES[args.model_type]
    model_kwargs = {'num_classes': task_classes[next(iter(task_classes))]}
    if args.model_type in ['mlp', 'vit']:
        model_kwargs.update({'win_len': args.win_len, 'feature_size': args.feature_size})
    if args.model_type == 'resnet18':
        model_kwargs.update({'in_channels': args.in_channels})
    if args.model_type == 'lstm':
        model_kwargs.update({'feature_size': args.feature_size})
    if args.model_type == 'vit':
        model_kwargs.update({'emb_dim': args.emb_dim, 'dropout': args.dropout})
    if args.model_type == 'transformer':
        model_kwargs.update({'feature_size': args.feature_size, 'd_model': args.emb_dim, 'dropout': args.dropout})

    print(f"Creating {args.model_type} backbone model...")
    backbone_model = ModelClass(**model_kwargs)
    print(f"Backbone parameters: {sum(p.numel() for p in backbone_model.parameters())}")

    # Extract transformer embeddings
    if args.model_type == 'transformer':
        class TransformerEmbedding(nn.Module):
            def __init__(self, cls_model):
                super().__init__()
                self.input_proj = cls_model.input_proj
                self.pos_encoder = cls_model.pos_encoder
                self.transformer = cls_model.transformer

            def forward(self, x):
                x = x.squeeze(1)
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                x = self.transformer(x)
                return x.mean(dim=1)

        backbone = TransformerEmbedding(backbone_model)
    else:
        raise NotImplementedError("Multi-task adapter supports only transformer backbone currently.")

    # Attach config with both dict and attribute access
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"No such attribute: {name}")

    d_model = args.emb_dim
    num_layers = len(backbone_model.transformer.layers)
    backbone.config = ConfigDict(
        model_type="bert",
        hidden_size=d_model,
        num_attention_heads=8,
        num_hidden_layers=num_layers,
        intermediate_size=d_model * 4,
        hidden_dropout_prob=args.dropout,
        use_return_dict=False
    )

    # Multi-task adapter
    model = MultiTaskAdapterModel(
        backbone, task_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Optimizer & criterion
    opt_params = list(model.adapters.parameters()) + list(model.heads.parameters())
    optimizer = AdamW(opt_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = []
    best_val, no_improve, best_state = float('inf'), 0, None

    def evaluate(loader):
        model.eval()
        tot_loss = tot_correct = tot_n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                b = y.size(0)
                tot_loss += loss.item() * b
                tot_correct += (logits.argmax(1) == y).sum().item()
                tot_n += b
        return tot_loss / tot_n, tot_correct / tot_n

    for epoch in range(1, args.epochs + 1):
        model.train()
        for task, loader in train_loaders.items():
            model.set_active_task(task)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validation
        row = {'epoch': epoch}
        val_losses = []
        print(f"\nValidation after epoch {epoch}:")
        for task, vloader in val_loaders.items():
            model.set_active_task(task)
            v_l, v_a = evaluate(vloader)
            val_losses.append(v_l)
            row[f"{task}_val_loss"] = v_l
            row[f"{task}_val_acc"] = v_a
            print(f"  {task}: loss={v_l:.4f}, acc={v_a:.4f}")
            # Confusion matrix
            tm = TaskTrainer(
                model=model,
                train_loader=None,
                val_loader=vloader,
                test_loader=None,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                device=device,
                save_path=os.path.join(args.save_dir, 'results'),
                num_classes=task_classes[task],
                label_mapper=vloader.dataset.label_mapper,
                config=None
            )
            tm.plot_confusion_matrix(data_loader=vloader, epoch=epoch, mode='val')

        history.append(row)
        avg_val = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: avg_val_loss={avg_val:.4f}")

        # Early stopping
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            best_state = {
                'adapters': model.adapters.state_dict(),
                'heads': model.heads.state_dict()
            }
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state:
        model.adapters.load_state_dict(best_state['adapters'])
        model.heads.load_state_dict(best_state['heads'])

    # Save history
    res_dir = os.path.join(args.save_dir, 'results')
    os.makedirs(res_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(res_dir, 'multitask_val.csv'), index=False)

    # Final test evaluation
    print("Final test results:")
    for task, tdict in test_loaders.items():
        for split, tloader in tdict.items():
            model.set_active_task(task)
            t_l, t_a = evaluate(tloader)
            print(f"{task} [{split}]: loss={t_l:.4f}, acc={t_a:.4f}")
            tm = TaskTrainer(
                model=model,
                train_loader=None,
                val_loader=tloader,
                test_loader=None,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                device=device,
                save_path=res_dir,
                num_classes=task_classes[task],
                label_mapper=tloader.dataset.label_mapper,
                config=None
            )
            tm.plot_confusion_matrix(data_loader=tloader, epoch=None, mode=split)

    # Save adapters + heads
    torch.save(
        {'adapters': model.adapters.state_dict(), 'heads': model.heads.state_dict()},
        os.path.join(args.save_dir, 'multitask_adapters.pt')
    )


if __name__ == '__main__':
    # Call main without arguments
    main()
