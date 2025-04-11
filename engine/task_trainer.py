import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR,ReduceLROnPlateau
from util import save_checkpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy

def warmup_schedule(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 1.0


def supervised_train(model, supervised_support_loader, supervised_test_loader, num_epochs_2,num_classes,
                learning_rate, task_criterion, device, optimizer_decay_rate, patience, save_path):
    model = model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=optimizer_decay_rate)

    # Warmup Scheduler
    warmup_epochs = 5
    # scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # Regular scheduler after warmup
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
    #                                  weight_decay=optimizer_decay_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=optimizer_decay_rate)

    l = []
    warmup_epochs = 5
    transfer_epochs = 10
    lowest_val_loss = 100

    lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Records for tracking progress
    supervised_records = []
    train_losses = []
    val_losses = []
    total_train_loss = 0

    for epoch in range(num_epochs_2):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        train_all_true_labels = []
        train_all_pred_labels = []

        val_all_true_labels = []
        val_all_pred_labels = []

        for batch_idx, (inputs, labels) in enumerate(supervised_support_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes).float()

            # Calculate the time requried for each data piece
            start_time = time.time()

            optimizer.zero_grad()
            outputs = model(inputs, flag="supervised")
            loss = task_criterion(outputs, labels_one_hot)
            loss.backward()
            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            elapsed_time = time.time() - start_time
            time_per_input = elapsed_time / inputs.size(0)

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += torch.eq(predict_y, labels).float().mean().item()
            train_true_labels = labels.cpu().numpy()
            train_pred_labels = predict_y.cpu().numpy()
            train_all_true_labels.extend(train_true_labels)
            train_all_pred_labels.extend(train_pred_labels)
            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(supervised_support_loader)
            train_losses.append(avg_train_loss)

        epoch_loss /= len(supervised_support_loader.dataset)
        epoch_accuracy /= len(supervised_support_loader)
        print('Epoch:{}, Accuracy:{:.4f}, Loss:{:.9f}'.format(epoch + 1, epoch_accuracy, epoch_loss))
        print('Epoch:{}, Time per input during training:{:.4f} s'.format(epoch + 1, time_per_input))

          # Reset learning rate, optimizer, and scheduler after warmup period
        # if epoch == transfer_epochs:
        #   # learning_rate = 0.001  # Adjust to your desired learning rate
        #   # Unfreeze all layers
        #     for param in model.parameters():
        #         param.requires_grad = True
        #   # Redefine optimizer and scheduler to include all parameters
        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=optimizer_decay_rate)
        #     lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
        #     scheduler = LambdaLR(optimizer, lr_lambda)

        #     # Update the scheduler
        #     if epoch < warmup_epochs:
        #         scheduler_warmup.step()
        # else:
        #     scheduler.step()
        # scheduler.step(avg_val_loss)
        scheduler.step()

        # Add validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        val_accuracies = []
        with torch.no_grad():
            for inputs, labels in supervised_test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes).float()
                outputs = model(inputs,flag="supervised")
                loss = task_criterion(outputs, labels_one_hot)
                val_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs, dim=1)
                val_accuracy += torch.eq(predict_y, labels).float().mean().item()
                val_true_labels = labels.cpu().numpy()
                val_pred_labels = predict_y.cpu().numpy()
                val_all_true_labels.extend(val_true_labels)
                val_all_pred_labels.extend(val_pred_labels)

        val_loss /= len(supervised_test_loader.dataset)
        val_accuracy /= len(supervised_test_loader)
        print('Val Accuracy:{:.4f}, Val Loss:{:.9f}'.format(val_accuracy, val_loss))
        cur_res = {'Epochs': epoch,
                   'Validation Accuracy': val_accuracy,
                   'Validation Loss': val_loss,
                   'Train Accuracy': epoch_accuracy,
                   'Train Loss': epoch_loss }
        l.append(cur_res)
        # if val_loss <= lowest_val_loss:
        #   lowest_val_loss =val_loss
        #   checkpoint(model, save_path+"_best_model.pth")
        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                model.load_state_dict(best_model)
                break
        torch.save(model.state_dict(), f"{save_path}_supervised_model_best.pth")

    # print('Starting supervised training phase.')
    # for epoch in range(num_epochs_2):
    #     model.train()
    #     total_train_loss = 0
    #     for data in supervised_support_loader:
    #         optimizer.zero_grad()
    #         x, y = data
    #         x = x.to(device)
    #         y = y.to(device)
    #         y_one_hot = torch.nn.functional.one_hot(y, num_classes).float()
    #         # Forward pass through the model
    #         outputs = model(x, flag='supervised')
    #         loss = task_criterion(outputs, y_one_hot)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_train_loss += loss.item()
    #
    #     avg_train_loss = total_train_loss / len(supervised_support_loader)
    #     train_losses.append(avg_train_loss)
    #     x, _ = next(iter(supervised_support_loader))  # Get a batch of input data



        # # Validation phase
        # model.eval()
        # total_val_loss = 0
        # correct_predictions = 0
        # total_samples = 0
        # val_accuracies = []
        # with torch.no_grad():
        #     for data in supervised_test_loader:
        #         x, y = data
        #         x = x.to(device)
        #         y = y.to(device)
        #         y_one_hot = torch.nn.functional.one_hot(y, num_classes).float()
        #         outputs = model(x, flag='supervised')
        #         loss = task_criterion(outputs, y_one_hot)
        #         total_val_loss += loss.item()
        #
        #         _, predicted = torch.max(outputs.data, 1)
        #         correct_predictions += (predicted == y).sum().item()
        #         total_samples += y.size(0)
        #
        # avg_val_loss = total_val_loss / len(supervised_test_loader)
        # val_losses.append(avg_val_loss)
        #
        # # Calculate accuracy
        # validation_accuracy = correct_predictions / total_samples
        # val_accuracies.append(validation_accuracy)  # If you're tracking accuracy over epochs

        # print(
        #     f'Epoch {epoch + 1}/{num_epochs_2}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')
        # # Early stopping and checkpointing
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     # Save the best model
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_val_loss,
        #     }, filename=f"{save_path}_supervised_model_best.pth")
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f'Early stopping triggered after {patience} epochs without improvement')
        #         break

    ## Added the vlisualization of training and validation confusion matrixs
    train_cm = confusion_matrix(train_all_true_labels, train_all_pred_labels)
    val_cm = confusion_matrix(val_all_true_labels, val_all_pred_labels)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ConfusionMatrixDisplay(train_cm).plot(ax=ax[0])
    ax[0].set_title('Training Confusion Matrix')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    ConfusionMatrixDisplay(val_cm).plot(ax=ax[1])
    ax[1].set_title('Validation Confusion Matrix')
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')
    plt.show()
    fig.savefig(save_path+"_train_confusionmat_unnormalized.png")


    # Plotting training and validation loss
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=epochs, y=train_losses, label='Training Loss', ax=ax)
    sns.lineplot(x=epochs, y=val_losses, label='Validation Loss', ax=ax)
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

    # Save the figure
    fig.savefig(save_path + "supervised_train_validation_loss.png")

    # Plot accuracy
    ax2 = ax.twinx()
    sns.lineplot(x=epochs, y=val_accuracies, label='Validation Accuracy', ax=ax2, color='r')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')

    plt.show()

    # Save the figure
    fig.savefig(save_path + "supervised_train_validation_loss_accuracy.png")

    return model, supervised_records


def subnet():
    return