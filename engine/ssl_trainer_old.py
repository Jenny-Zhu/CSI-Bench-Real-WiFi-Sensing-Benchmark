import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from util import save_checkpoint


import torch
import matplotlib.pyplot as plt
import seaborn as sns

def unsupervised_train_old(model, unsupervised_train_loader, num_epochs_1, learning_rate, criterion, device, augmentor, patience, save_path):
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
    # augmentation = augmentor.apply_augmentations()


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

        # print("Dataset length:", len(unsupervised_train_loader.dataset))

        for data in unsupervised_train_loader:
            x = data  # Ignore labels
            x = x.to(device)

            # Apply augmentation to each image in the batch to get x1, x2
            x1 = augmentor.apply_augmentations(x)
            x2 = augmentor.apply_augmentations(x)
            x1 = x1.to(device)
            x2 = x2.to(device)

            # Forward pass through the model
            z1 = model(x1)
            z2 = model(x2)
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)

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
    plt.plot(range(1, num_epochs_1 + 1), learning_rates, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    plt.show()

    return model, unsupervised_records