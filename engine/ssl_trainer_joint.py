import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingWarmRestarts, LambdaLR
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from util import save_checkpoint
from util.mask_creater import create_mask


# ------------------------------
#        Mutual Info Functions
# ------------------------------

def calculate_pairwise_mi(z1, z2):
    """
    Computes pairwise 1D mutual information (MI) between corresponding features
    of z1 and z2, using sklearn's mutual_info_regression.

    Args:
        z1 (ndarray): shape (N, D)
        z2 (ndarray): shape (N, D)
    Returns:
        mi_scores (ndarray): shape (D,), the MI for each feature dimension.
    """
    n_features = z1.shape[1]
    mi_scores = np.zeros(n_features)
    for i in range(n_features):
        # 'mutual_info_regression' requires 2D shape for X
        mi_scores[i] = mutual_info_regression(z1[:, i].reshape(-1, 1), z2[:, i])
    return mi_scores


def calculate_mi(x, z1, z2):
    """
    Calculates pairwise MI among x->z1, x->z2, and z1->z2
    after PCA reduction.

    This function only processes the first 4 samples of each
    input (due to x[0:4]), presumably for debug or memory reasons.

    Args:
        x  (torch.Tensor): shape (B, ...)
        z1 (torch.Tensor): shape (B, ...)
        z2 (torch.Tensor): shape (B, ...)
    Returns:
        (mi_X_Z1, mi_X_Z2, mi_Z1_Z2): floats, average MI
    """
    # Keep only first 4 samples
    x  = x[:4].cpu()
    z1 = z1[:4].cpu()
    z2 = z2[:4].cpu()

    batch_size = x.shape[0]

    # Flatten
    x_flat  = x.reshape(batch_size, -1).detach().numpy()
    z1_flat = z1.reshape(batch_size, -1).detach().numpy()
    z2_flat = z2.reshape(batch_size, -1).detach().numpy()

    # PCA constraints
    max_components = min(batch_size, z1_flat.shape[1])
    pca = PCA(n_components=min(10, max_components))

    # Transform
    x_reduced  = pca.fit_transform(x_flat)
    z1_reduced = pca.fit_transform(z1_flat)
    z2_reduced = pca.fit_transform(z2_flat)

    mi_X_Z1   = np.mean(calculate_pairwise_mi(x_reduced, z1_reduced))
    mi_X_Z2   = np.mean(calculate_pairwise_mi(x_reduced, z2_reduced))
    mi_Z1_Z2  = np.mean(calculate_pairwise_mi(z1_reduced, z2_reduced))

    return mi_X_Z1, mi_X_Z2, mi_Z1_Z2


# ------------------------------
#       Training Functions
# ------------------------------

def unsupervised_train(
    model,
    unsupervised_train_loader,
    num_epochs_1,
    learning_rate,
    criterion,
    device,
    augmentor,
    patience,
    save_path
):
    """
    Self-supervised training loop for an unspecified model that
    outputs (z1_proj, z2_proj, reconstructed).

    Args:
        model: PyTorch model
        unsupervised_train_loader: DataLoader providing unlabeled data
        num_epochs_1 (int): number of training epochs
        learning_rate (float)
        criterion: e.g. NT-Xent Loss
        device: torch.device
        augmentor: data augmentation function
        patience (int): early stopping patience
        save_path (str): directory to save best model, logs, etc.

    Returns:
        (model, records): updated model and any record object
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Cosine Annealing Restarts
    T_0 = 10
    scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-4)

    # Warmup scheduler
    warmup_epochs = 5
    warmup_lambda = lambda epoch: min((epoch + 1) / warmup_epochs, 1.0)
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    learning_rates = []
    unsupervised_records = []
    mi_records = []  # In case you want to store MI data

    print('Starting self-supervised training phase...')
    for epoch in range(num_epochs_1):
        # Pick scheduler
        scheduler = scheduler_warmup if epoch < warmup_epochs else scheduler_cosine

        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        print(f'[Epoch {epoch+1}/{num_epochs_1}] Current LR: {current_lr:.2e}')

        model.train()
        total_loss = 0.0

        for data in unsupervised_train_loader:
            x = data.to(device)

            # Create mask
            mask = create_mask(
                batch_size=x.shape[0],
                seq_len=x.shape[1],
                feature_size=x.shape[2],
                row_mask_ratio=0.1,
                col_mask_ratio=0.1
            ).to(device)

            # Two augmented views
            x1 = augmentor.apply_augmentations(x)
            x2 = augmentor.apply_augmentations(x)

            # Forward pass
            z1_proj, z2_proj, reconstructed = model(x1, x2, mask=mask, flag='joint')

            # Loss
            contrastive_loss = criterion(z1_proj, z2_proj)
            reconstruction_loss = F.mse_loss(reconstructed, x)
            loss = contrastive_loss + 1.0 * reconstruction_loss  # lambda_recon = 1.0

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # average train loss
        avg_train_loss = total_loss / len(unsupervised_train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')

        # Step the chosen scheduler
        scheduler.step()

        # Early stopping check
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_path, f"best_model_epoch_{epoch+1}.pth"))
        else:
            epochs_no_improve += 1

        print(f'Epochs without improvement: {epochs_no_improve}, Best Loss: {best_loss:.4f}')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

    # Plot training curve
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses).set_title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_path, "training_loss_plot.png"))
    plt.show()

    # Optionally, store MI records
    mi_records_df = pd.DataFrame.from_records(mi_records)
    mi_records_df.to_csv(os.path.join(save_path, "mi_records.csv"), index=False)

    return model, unsupervised_records


def unsupervised_train_acf(
    model,
    unsupervised_train_loader,
    num_epochs_1,
    learning_rate,
    criterion,
    device,
    augmentor,
    patience,
    save_path
):
    """
    Similar to unsupervised_train but uses a different scheduling approach
    (SequentialLR of warmup + CosineAnnealingWarmRestarts).
    Also uses slightly different data shape assumptions
    (with x.shape[-2], x.shape[-1]) for mask creation.

    Args:
        model: PyTorch model
        unsupervised_train_loader: DataLoader
        num_epochs_1: int
        learning_rate: float
        criterion: e.g. NT-Xent
        device: torch.device
        augmentor: function or object (data augmentation)
        patience: int (early stopping patience)
        save_path: str, location to save model & logs

    Returns:
        (model, unsupervised_records)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Warmup + CosineAnnealingWarmRestarts using SequentialLR
    warmup_epochs = 5
    T_0 = 10

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-5)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    learning_rates = []
    unsupervised_records = []

    print('Starting self-supervised (ACF) training phase...')
    for epoch in range(num_epochs_1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(unsupervised_train_loader):
            x = batch.to(device)

            # shape-based mask
            mask = create_mask(
                batch_size=x.shape[0],
                seq_len=x.shape[-2],
                feature_size=x.shape[-1],
                row_mask_ratio=0.1,
                col_mask_ratio=0.1
            ).to(device)

            # Two augmented views
            x1 = torch.stack([augmentor(img) for img in x]).to(device)
            x2 = torch.stack([augmentor(img) for img in x]).to(device)

            # Model forward
            z1_proj, z2_proj, reconstructed = model(x1, x2, mask, flag='joint')

            # Loss
            contrastive_loss = criterion(z1_proj, z2_proj)
            reconstruction_loss = F.mse_loss(reconstructed, x.squeeze(1))
            loss = contrastive_loss + reconstruction_loss  # you might tune these weights

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            print(f'Batch {batch_idx} processed.')

        # End of epoch
        avg_train_loss = total_loss / len(unsupervised_train_loader)
        train_losses.append(avg_train_loss)

        # Step the combined scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f'[Epoch {epoch+1}/{num_epochs_1}] Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}')

        # Early stopping check
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, filename=os.path.join(save_path, "best_model_checkpoint_ssl.pth.tar"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

    # Plot training loss
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=epochs_range, y=train_losses, label='Training Loss', ax=ax)
    ax.set_title('Training Loss (ACF)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()
    fig.savefig(os.path.join(save_path, "train_loss_acf.png"))

    # Plot learning rate progression
    plt.figure(figsize=(10, 4))
    plt.plot(epochs_range, learning_rates, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    plt.show()

    return model, unsupervised_records
