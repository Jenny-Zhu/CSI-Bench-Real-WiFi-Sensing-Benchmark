from util import save_checkpoint
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
def calculate_pairwise_mi(z1, z2):
    n_features = z1.shape[1]
    mi_scores = np.zeros(n_features)
    for i in range(n_features):
        mi_scores[i] = mutual_info_regression(z1[:, i].reshape(-1, 1), z2[:, i])
    return mi_scores


def calculate_mi(x, z1, y):
    # Move tensors to CPU
    x = x[0:4].cpu()
    z1 = z1[0:4].cpu()
    y = y[0:4].cpu()

    # Get the batch size
    BATCH_SIZE = x.shape[0]
    # BATCH_SIZE = 1
    # Flatten the tensors
    z1_flat = z1.reshape(BATCH_SIZE, -1)
    y_flat = y.reshape(BATCH_SIZE, -1)
    x_flat = x.reshape(BATCH_SIZE, -1)

    # Detach the tensors before converting to numpy arrays for PCA
    z1_flat_np = z1_flat.detach().numpy()
    y_flat_np = y_flat.detach().numpy()
    x_flat_np = x_flat.detach().numpy()

    # Determine the maximum allowable number of components
    max_components = min(BATCH_SIZE, z1_flat_np.shape[1])

    # Initialize PCA with a feasible number of components
    pca = PCA(n_components=min(10, max_components))  # Ensuring it does not exceed the limit

    # Fit and transform the flat numpy arrays using PCA
    z1_reduced = pca.fit_transform(z1_flat_np)
    y_reduced = pca.fit_transform(y_flat_np)
    x_reduced = pca.fit_transform(x_flat_np)

    mi_X_Z1 = calculate_pairwise_mi(x_reduced, z1_reduced)
    mi_X_Z1 = np.mean(mi_X_Z1)
    mi_Y_Z1 = calculate_pairwise_mi(y_reduced, z1_reduced)
    mi_Y_Z1 = np.mean(mi_Y_Z1)
    mi_X_Y = calculate_pairwise_mi(x_reduced, y_reduced)
    mi_X_Y = np.mean(mi_X_Y)

    return mi_X_Z1, mi_Y_Z1, mi_Z1_Z2


def calculate_mi(x, z1, z2):
    # Move tensors to CPU
    x = x[0:4].cpu()
    z1 = z1[0:4].cpu()
    z2 = z2[0:4].cpu()

    # Get the batch size
    BATCH_SIZE = x.shape[0]
    # BATCH_SIZE = 1
    # Flatten the tensors
    z1_flat = z1.reshape(BATCH_SIZE, -1)
    z2_flat = z2.reshape(BATCH_SIZE, -1)
    x_flat = x.reshape(BATCH_SIZE, -1)

    # Detach the tensors before converting to numpy arrays for PCA
    z1_flat_np = z1_flat.detach().numpy()
    z2_flat_np = z2_flat.detach().numpy()
    x_flat_np = x_flat.detach().numpy()

    # Determine the maximum allowable number of components
    max_components = min(BATCH_SIZE, z1_flat_np.shape[1])

    # Initialize PCA with a feasible number of components
    pca = PCA(n_components=min(10, max_components))  # Ensuring it does not exceed the limit

    # Fit and transform the flat numpy arrays using PCA
    z1_reduced = pca.fit_transform(z1_flat_np)
    z2_reduced = pca.fit_transform(z2_flat_np)
    x_reduced = pca.fit_transform(x_flat_np)

    mi_X_Z1 = calculate_pairwise_mi(x_reduced, z1_reduced)
    mi_X_Z1 = np.mean(mi_X_Z1)
    mi_X_Z2 = calculate_pairwise_mi(x_reduced, z2_reduced)
    mi_X_Z2 = np.mean(mi_X_Z2)
    mi_Z1_Z2 = calculate_pairwise_mi(z1_reduced, z2_reduced)
    mi_Z1_Z2 = np.mean(mi_Z1_Z2)

    return mi_X_Z1, mi_X_Z2, mi_Z1_Z2

def unsupervised_train(model, unsupervised_train_loader, num_epochs_1, learning_rate, criterion, device, augmentor, patience, save_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Implementing SGDR with Cosine Annealing
    T_0 = 10  # Initial number of epochs in the first cycle before restart
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-4)

    # Implementing Learning Rate Warmup
    warmup_epochs = 5
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1))

    best_loss = float('inf')
    epochs_no_improve = 0
    mi_records = []
    unsupervised_records = []
    train_losses = []
    learning_rates = []

    print('Starting self-supervised training phase.')
    for epoch in range(num_epochs_1):
        scheduler = scheduler_warmup if epoch < warmup_epochs else scheduler_cosine
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        print(f'Epoch {epoch+1}, Current LR: {current_lr}')
        model.train()
        total_loss = 0

        for data in unsupervised_train_loader:
            x = data.to(device)  # Transfer data to the device once

            # Apply augmentation to each image in the batch to get x1, x2
            x1 = augmentor.apply_augmentations(x)
            x2 = augmentor.apply_augmentations(x)
            # print(f"x1 size: {x1.shape}")

            # Forward pass through the model
            z1, z2 = model(x1), model(x2)
            z1, z2 = torch.nn.functional.normalize(z1, p=2, dim=1), torch.nn.functional.normalize(z2, p=2, dim=1)

            # Calculate contrastive loss
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(unsupervised_train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{num_epochs_1}, Train Loss: {avg_train_loss:.4f}')

        # Update scheduler and checkpointing
        scheduler.step()
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path + f"best_model_epoch_{epoch+1}.pth")
        else:
            epochs_no_improve += 1

        print(f'Epochs without improvement: {epochs_no_improve}. Best loss: {best_loss}')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break

    # Visualization of training results
    sns.lineplot(x=list(range(1, len(train_losses) + 1)), y=train_losses).set_title('Training Loss over Epochs')
    plt.savefig(save_path + "training_loss_plot.png")
    plt.show()
    mi_records = pd.DataFrame.from_records(mi_records)
    mi_records.to_csv(save_path+"mi_records.csv")

    return model, unsupervised_records

def unsupervised_train_acf(model, unsupervised_train_loader, num_epochs_1, learning_rate, criterion, device, augmentor, patience, save_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # Implementing SGDR with Cosine Annealing
    T_0 = 10  # Initial number of epochs in the first cycle before restart
    scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-4)

    # Implementing Learning Rate Warmup
    warmup_epochs = 5
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1))

    # Early stopping and model checkpointing
    best_loss = float('inf')
    epochs_no_improve = 0

    # Records for tracking progress
    unsupervised_records = []
    train_losses = []
    learning_rates = []
    # Define the image size for augmentations
    augmentation = augmentor.get_augmentation()
    mi_record = []


    print('Starting self-supervised training phase.')
    for epoch in range(num_epochs_1):
        if epoch < warmup_epochs:
            scheduler = scheduler_warmup
        else:
            scheduler = scheduler_cosine
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        print(f'Epoch {epoch+1}, Current LR: {current_lr}')
        model.train()
        total_loss = 0
        total_mi_X_Z1 = 0
        total_mi_X_Z2 = 0
        total_mi_Z1_Z2 = 0
        for data in unsupervised_train_loader:
            x = data  # Ignore labels
            x = x.to(device)

            # Apply augmentation to each image in the batch to get x1, x2
            x1 = torch.stack([augmentation(img) for img in x]).to(device)
            x2 = torch.stack([augmentation(img) for img in x]).to(device)

            # Forward pass through the model
            z1 = model(x1)
            z2 = model(x2)
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
            # mi_X_Z1,mi_X_Z2,mi_Z1_Z2 = calculate_mi(x,z1,z2)
            # total_mi_X_Z1 = total_mi_X_Z1+mi_X_Z1
            # total_mi_X_Z2 = total_mi_X_Z2+mi_X_Z2
            # total_mi_Z1_Z2 = total_mi_Z1_Z2+mi_Z1_Z2

            # Calculate contrastive loss
            loss = criterion(z1, z2)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(unsupervised_train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch+1}/{num_epochs_1}, Train Loss: {avg_train_loss:.4f}')
        # average_mi_X_Z1 = total_mi_X_Z1 / len(unsupervised_train_loader)
        # average_mi_X_Z2 = total_mi_X_Z2 / len(unsupervised_train_loader)
        # average_mi_Z1_Z2 = total_mi_Z1_Z2 / len(unsupervised_train_loader)
        # print(f"Epoch {epoch + 1}, Average MI X-Z1: {average_mi_X_Z1}")
        # print(f"Epoch {epoch+1}, Average MI Z1-Z2: {average_mi_Z1_Z2}")
        # mi_res = {'Epochs:': epoch,
        #           'mi_X_Z1': average_mi_X_Z1,
        #           'mi_X_Z2': average_mi_X_Z2,
        #           'mi_Z1_Z2': average_mi_Z1_Z2}
        # mi_record.append(mi_res)

        # Update scheduler after each epoch
        if epoch < warmup_epochs:
            scheduler_warmup.step()  # Warmup phase
        else:
            scheduler_cosine.step(epoch - warmup_epochs)  # Adjust based on the total epochs minus warmup epochs

        # Check for early stopping
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            # Save the best model
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, filename=save_path + "best_model_checkpoint_ssl.pth.tar")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break

    # After training, plot the training loss
    import matplotlib.pyplot as plt
    import seaborn as sns

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=epochs, y=train_losses, label='Training Loss', ax=ax)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

    # Save the figure
    fig.savefig(save_path + "train_loss.png")

    # Plotting learning rate progression
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), learning_rates, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    plt.show()
    # mi_records = pd.DataFrame.from_records(mi_record)
    # mi_records.to_csv(save_path+"mi_records.csv")

    return model, unsupervised_records