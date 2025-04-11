import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time
from torch.optim.lr_scheduler import LambdaLR
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
import torch.optim as optim

def calculate_pairwise_mi_y(z1, y):
    # This function now assumes y is categorical and z1 can be one or more features.
    n_features = z1.shape[1]
    mi_scores = np.zeros(n_features)
    y = y.ravel()
    for i in range(n_features):
        mi_scores[i] = mutual_info_classif(z1[:, i].reshape(-1, 1), y)
    return mi_scores

def calculate_pairwise_mi(x, z1):
    n_features = x.shape[1]
    mi_scores = np.zeros(n_features)
    for i in range(n_features):
        # Ensure the input feature and the target are in the correct shapes
        feature_x = x[:, i].reshape(-1, 1)  # Feature from x
        target_z1 = z1[:, i].ravel()  # Target from z1 must be 1D

        # Calculate mutual information, ensuring target_z1 is 1D
        mi_scores[i] = mutual_info_regression(feature_x, target_z1)
    return mi_scores

def calculate_mi(x, z1, y):
    # Move tensors to CPU
    x = x.cpu()
    z1 = z1.cpu()
    y = y.cpu()

    # Flatten the tensors
    BATCH_SIZE = x.shape[0]
    z1_flat = z1.reshape(BATCH_SIZE, -1)
    x_flat = x.reshape(BATCH_SIZE, -1)

    # Detach the tensors before converting to numpy arrays
    z1_flat_np = z1_flat.detach().numpy()
    x_flat_np = x_flat.detach().numpy()
    y_np = y.detach().numpy()  # No need to flatten y as it's categorical
    # Determine the maximum allowable number of components for PCA
    max_components = min(BATCH_SIZE, z1_flat_np.shape[1])

    # Initialize PCA with a feasible number of components
    pca = PCA(n_components=min(10, max_components))

    # Fit and transform the flat numpy arrays using PCA
    z1_reduced = pca.fit_transform(z1_flat_np)
    x_reduced = pca.fit_transform(x_flat_np)
    # x_reduced = x_reduced.ravel()
    # z1_reduced = z1_reduced.ravel()
    # Calculate pairwise mutual information
    mi_X_Z1 = calculate_pairwise_mi(x_reduced, z1_reduced)
    mi_X_Z1 = np.mean(mi_X_Z1)

    mi_Y_Z1 = calculate_pairwise_mi_y(z1_reduced, y_np)
    mi_Y_Z1 = np.mean(mi_Y_Z1)

    mi_X_Y = calculate_pairwise_mi_y(x_reduced, y_np)
    mi_X_Y = np.mean(mi_X_Y)

    return mi_X_Z1, mi_Y_Z1, mi_X_Y

def checkpoint(model, filename):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the model state
        torch.save(model.state_dict(), filename)
        print(f"Model saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save the model to {filename}. Error: {e}")
# def warmup_schedule(epoch, warmup_epochs):
#     if epoch < warmup_epochs:
#         return epoch / warmup_epochs
#     else:
#         return 1.0

def warmup_cosine_lambda(epoch, warmup_epochs, transfer_epochs, total_epochs,
                         unfreeze_scale=0.1):
    if epoch < warmup_epochs:
        # linear ramp up from near 0 to 1
        return float(epoch + 1) / warmup_epochs
    elif epoch < transfer_epochs:
        # constant 1.0 (no scaling)
        return 1.0
    else:
        # Cosine from transfer_epochs..total_epochs
        progress = float(epoch - transfer_epochs) / float(total_epochs - transfer_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return unfreeze_scale * cosine_factor

def piecewise_lr_lambda(epoch, warmup_epochs, transfer_epochs, total_epochs):
    """
    Returns a multiplicative factor for the base LR at a given epoch.

    Phase 1: Warmup from epoch 0 -> warmup_epochs
    Phase 2: Constant LR from warmup_epochs -> transfer_epochs
    Phase 3: Cosine decay from transfer_epochs -> total_epochs
    """
    if epoch < warmup_epochs:
        # Linearly scale from near 0 to 1
        return float(epoch + 1) / warmup_epochs
    elif epoch < transfer_epochs:
        # Keep LR factor = 1.0 (no scaling)
        return 1.0
    else:
        # Cosine decay from transfer_epochs..total_epochs
        progress = float(epoch - transfer_epochs) / float(total_epochs - transfer_epochs)
        # Standard cosine factor from 1.0 down to 0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_factor


def freeze_first_n_layers(model, n=3):
    """
    Example utility to freeze the first `n` layers of a model.
    Adjust logic to match your model’s layer naming or structure.
    """
    layer_count = 0
    for name, param in model.named_parameters():
        # e.g., if you have a typical CNN, you might check name starts with 'conv1', 'layer1', etc.
        # This is model-specific; adapt as needed.
        if layer_count < n:
            param.requires_grad = False
        else:
            param.requires_grad = True
        layer_count += 1


def supervised_train_acf(model, train_loader, valid_loader, device, criterion,save_path,
                num_epochs=50,
                learning_rate=1e-3,
                optimizer_decay_rate=1e-5,
                weight_setting_string='none',
                patience=5,
                IF_TRANSFER=0,
                warmup_epochs=10,
                transfer_epochs=20):
    model = model.to(device)
    filename = os.path.join(save_path, f"best_model.pth.tar")

    # -----------------------------------------------------------------
    # 1) Optionally freeze the first N layers for transfer learning
    # -----------------------------------------------------------------
    if IF_TRANSFER == 1:
        # for name, param in model.named_parameters():
            # if "freq_cnn.conv1" in name or "freq_cnn.conv2" in name:
            #     assert not param.requires_grad, f"{name} should be frozen!"
            # if "encoder.layers.0" in name or "encoder.layers.1" in name or "encoder.layers.2" in name:
            #     assert not param.requires_grad, f"{name} should be frozen!"
            # if "classifier" in name:
            #     assert param.requires_grad, "Classifier should be trainable!"
        print("All specified layers are frozen ✅")
        print("[INFO] Freezing first 3 layers for transfer learning ...")
        # Only train the remaining (unfrozen) parameters.
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=learning_rate,
                                     weight_decay=optimizer_decay_rate)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0

    else:
        print("[INFO] Training all layers from scratch ...")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=optimizer_decay_rate)

    # Warmup schedule (if you want to adjust LR for first `warmup_epochs`):
    def warmup_schedule(epoch, warmup_epochs):
        # Example: linearly ramp up from 0.1*LR to LR over warmup_epochs
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 1.0

    # Initialize scheduler
    lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_model_state = None

    # For early stopping
    epochs_no_improve = 0

    # For logging epoch results
    epoch_records = []

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        start_epoch_time = time.time()


        train_all_true_labels = []
        train_all_pred_labels = []
        epoch_loss = 0.0
        epoch_accuracy = 0

        # ------------- Training -------------
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # # # Example batch-wise normalization (often do in dataset transforms)
            # mean = inputs.mean(dim=[0,1,2], keepdim=True)
            # std = inputs.std(dim=[0,1,2], keepdim=True)
            # inputs = (inputs - mean) / (std + 1e-6)
            start_time = time.time()
            optimizer.zero_grad()
            outputs, z = model(inputs, flag="supervised")
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
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

        epoch_loss /= len(train_loader.dataset)
        epoch_accuracy /= len(train_loader)
        print('Epoch:{}, Accuracy:{:.4f}, Loss:{:.9f}'.format(epoch + 1, epoch_accuracy, epoch_loss))
        print('Epoch:{}, Time per input during training:{:.4f} s'.format(epoch + 1, time_per_input))

        # ------------- Unfreeze logic after transfer_epochs -------------
        if IF_TRANSFER == 1 and epoch == transfer_epochs:
            print(f"[INFO] Unfreezing all layers at epoch {epoch + 1}...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=optimizer_decay_rate)
            lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
            scheduler = LambdaLR(optimizer, lr_lambda)

        scheduler.step()

        # ------------- Validation loop -------------

        model.eval()
        val_all_true_labels = []
        val_all_pred_labels = []
        val_loss = 0.0
        val_accuracy = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # # Example batch-wise normalization (often do in dataset transforms)
                # mean = inputs.mean(dim=[0, 1, 2], keepdim=True)
                # std = inputs.std(dim=[0, 1, 2], keepdim=True)
                # inputs = (inputs - mean) / (std + 1e-6)

                outputs, z = model(inputs, flag="supervised")
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs, dim=1)
                val_accuracy += torch.eq(predict_y, labels).float().mean().item()
                val_true_labels = labels.cpu().numpy()
                val_pred_labels = predict_y.cpu().numpy()
                val_all_true_labels.extend(val_true_labels)
                val_all_pred_labels.extend(val_pred_labels)

        val_loss /= len(valid_loader.dataset)
        val_accuracy /= len(valid_loader)
        print('Val Accuracy:{:.4f}, Val Loss:{:.9f}'.format(val_accuracy, val_loss))

        # ----------------- Logging -----------------
        epoch_record = {
            'Epoch': epoch + 1,
            'Train Loss': epoch_loss,
            'Train Accuracy': epoch_accuracy,
            'Val Loss': val_loss,
            'Val Accuracy': val_accuracy
        }
        epoch_records.append(epoch_record)

        # ----------------- Checkpoint & Early Stopping -----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_model_epoch=epoch+1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered!")
                break

        end_epoch_time = time.time()
        epoch_duration = end_epoch_time - start_epoch_time
        print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds.\n")

    # ----------------- Load best model before returning -----------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

        print(f'Best model saved at epoch {best_model_epoch}')


    train_cm = confusion_matrix(train_all_true_labels, train_all_pred_labels)
    val_cm = confusion_matrix(val_all_true_labels, val_all_pred_labels)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ConfusionMatrixDisplay(train_cm).plot(ax=ax[0])
    ax[0].set_title('Training Confusion Matrix')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    ConfusionMatrixDisplay(val_cm).plot(ax=ax[1])
    ax[1].set_title('Validation Confusion Matrix')
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')

    plt.tight_layout()
    plot_filename = os.path.join(save_path, "confusion_matrices.png")
    plt.savefig(plot_filename)
    plt.show()

    # Turn the epoch records into a DataFrame
    df = pd.DataFrame(epoch_records)
    return model, df





def supervised_test_acf(model, tensor_loader, criterion, device, num_classes,task,save_path):
    t= []

    model.to(device)
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    test_time_per_input = 0.0

    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in tensor_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Example batch-wise normalization (often do in dataset transforms)
            # mean = inputs.mean(dim=[0, 1, 2], keepdim=True)
            # std = inputs.std(dim=[0, 1, 2], keepdim=True)
            # inputs = (inputs - mean) / (std + 1e-6)
            # labels_one_hot = torch.nn.functional.one_hot(labels, num_classes).float()
            #Calculate the time required by each sample during testing
            start_time = time.time()
            outputs,z = model(inputs, flag="supervised")
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            elapsed_time = time.time() - start_time
            # Accumulate per-sample test time
            test_time_per_input += elapsed_time

            # Accumulate total correct predictions
            test_correct += (predict_y == labels).sum().item()

            # Accumulate number of samples
            test_total += labels.size(0)

            # Accumulate sum of losses
            test_loss_sum += loss.item() * inputs.size(0)

            # Save predictions/labels for confusion matrix
            y_pred.extend(predict_y.cpu().numpy())  # <- ensure CPU + numpy
            y_true.extend(labels.cpu().numpy())

        # Overall test loss / accuracy
        test_loss = test_loss_sum / test_total
        test_acc = test_correct / test_total

        # Average time per sample across the entire test set
        test_time_per_input /= test_total

        print("Test accuracy: {:.4f}, loss: {:.5f}".format(test_acc, test_loss))
        print("Test time required per sample: {:.4f}".format(test_time_per_input))
        print("len(tensor_loader.dataset): {}, len(tensor_loader): {}".format(len(tensor_loader.dataset),
                                                                              len(tensor_loader)))

    # constant for classes
    if task == 'FourClass':
      classes = ('Human', 'Pet', 'IRobot', 'Fan')
      labels = np.array([0, 1, 2, 3])  # assuming the classes are coded as 0,1,2,3
    elif task == 'HumanNonhuman':
      classes = ('Human','Nonhuman')
      labels = np.array([1, 0])  # assuming the classes are coded as 1,0
    elif task == 'ThreeClass':
      classes = ('Human', 'Pet', 'IRobot')
      labels = np.array([0, 1, 2])  # assuming the classes are coded as 0,1,2
    elif task =='NTUHumanID':
      classes = ('001', '002', '003', '004', '005',
               '006', '007', '008', '009', '010',
               '011', '012', '013', '014', '015')
      labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14])
    elif task =='NTUHAR':
      classes = ('run', 'walk', 'box', 'circle', 'clean', 'fall')
      labels = np.array([0, 1, 2, 3, 4, 5])
    elif task =='Widar':
      classes = ('PP', 'Sw', 'Cl', 'Sl', 'DNH', 'DOH','DRH','DTH',
                   'DZH','DZ','DN','DO','Dr1','Dr2','Dr3','Dr4','Dr5',
                   'Dr6','Dr7','Dr8','Dr9','Dr10')
      labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21])
    elif task == 'DetectionandClassification':
      classes = ('NoMotion','Human', 'Pet', 'IRobot','Fan')
      labels = np.array([0, 1, 2, 3, 4])  # assuming the classes are coded as 0,1,2
    elif task == 'Detection':
      classes = ('NoMotion','Motion')
      labels = np.array([0, 1])  # assuming the classes are coded as 0,1,2
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    cf_matrix = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    row_sums = np.sum(cf_matrix, axis=1)[:, None]
    normalized_cf_matrix = np.where(row_sums != 0, cf_matrix / row_sums, 0)
    # Create the DataFrame
    df_cm = pd.DataFrame(normalized_cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    fig_cm, ax1 = plt.subplots(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, ax = ax1)
    ax1.set_title('Normalized confusion matrix')
    ax1.set_xlabel('Predicted classes')
    ax1.set_ylabel('Actual classes')
    fig_cm.savefig(save_path+"_test_confusionmat_normalized.png")
    df_cm.to_csv(save_path+"_test_confusionmat_normalized.csv")
    if(df_cm.iloc[0,0]>=0.2 )and (df_cm.iloc[1,0]<0.08):
      if_save_model = True

    else:
      if_save_model = False
    df_cm_un = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    fig_cm_un, ax2 = plt.subplots(figsize = (12,7))
    sn.heatmap(df_cm_un, annot=True, ax = ax2)
    ax2.set_title('Unnormalized confusion matrix')
    ax2.set_xlabel('Prediected classes')
    ax2.set_ylabel('Actual classes')
    fig_cm_un.savefig(save_path+"_test_confusionmat_unnormalized.png")
    df_cm_un.to_csv(save_path + "_test_confusionmat_unnormalized.csv")
    pd_test_report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    pd_test_report.to_csv(save_path+"_test_report.csv")
    # disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
    #                                display_labels=["Human","Pet","iRobot","Fan"])
    # disp.plot()
    # plt.show()
    test_results = {'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Test time required per sample': test_time_per_input}
    t.append(test_results)
    df_test = pd.DataFrame.from_records(t)
    return df_test,if_save_model
