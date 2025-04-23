import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import engine.task_trainer as task_trainer
import engine.task_trainer_acf as task_trainer_acf
from load import load_data_supervised, load_model_pretrained,save_data_supervised,\
    load_acf_supervised,load_acf_unseen_environ,fine_tune_model,load_model_scratch, load_model_trained
from tools.loss_fuction import NtXentLoss,FocalLoss
from data.data_augmentation import DataAugmentation
from model import ViT_Parallel
from model import ViT_Parallel,ViT_MultiTask
import load
import copy

def freeze_cnn_layers(model, num_frozen_layers=3):
    cnn_layers = [model.freq_cnn.conv1, model.freq_cnn.conv2]  # Adjust based on your SmallFreqCNN
    for layer in cnn_layers[:num_frozen_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Froze first {num_frozen_layers} CNN layers.")
def freeze_encoder_layers(model, num_frozen_layers=3):
    for layer in model.encoder.layers[:num_frozen_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Froze first {num_frozen_layers} encoder layers.")

def freeze_input_embeddings(model):
    for param in model.input_embed.parameters():
        param.requires_grad = False
    print("Froze input embeddings.")


def print_frozen_status(model):
    for name, param in model.named_parameters():
        print(f"{name}: {'FROZEN' if not param.requires_grad else 'Trainable'}")


def run_task_acf():
    learning_rate = 1e-4
    win_len = 250
    feature_size = 98
    emb_size = int((win_len / 10) * (feature_size / 2))
    patience = 10
    num_epochs_2 = 300
    depth = 6
    in_channels = 1
    model_name = "ViT"
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    freeze_layer_cnn = 2
    freeze_layer_encoder = 6

    checkpoint_path = ".\\experiments\\test\\ssl\\02122025\\CSI100_DataAug_wifi_joint_2d_rel_variant\\ViT\\best_model_checkpoint_ssl.pth.tar"
    results_dir = ".\\experiments\\02132025\\task\\OWHM3\\CSI100_DataAug_wifi_joint_2d_rel_variant_f6\\"
    BATCH_SIZE = 128
    DECAY_RATE = 1e-9
    IF_TRANSFER = 1
    task = 'ThreeClass'
    data_dir = ".\\dataset\\task\\HM3\\CSIMAT100\\train\\train"
    if IF_TRANSFER == 1:
        supervised_model = ViT_MultiTask(
        emb_dim=128,
        encoder_heads=4,
        encoder_layers=6,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        # max_dist=32,  # for relative time
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0,
        num_classes=num_classes,
        c_out=60,
        freq_out=6,  # The final freq dimension we want in the CNN
        max_len=512  # or some large number for variable T
        )
        state_dict = torch.load(checkpoint_path)
        pretrained_model_weights = state_dict["model_state_dict"]

        # Keys to load (exclude classifier and positional embedding)
        keys_to_load = [
            k for k in pretrained_model_weights.keys()
            if not k.startswith("classifier.")  # Skip final classifier
               and not k.startswith("pos_time.")  # Skip absolute pos embedding (you removed it)
               and not k.startswith("reconstruction_transformer.")  # Skip if not needed
               and not k.startswith("contrastive_head.")  # Optional: Skip if not using contrastive
               and not k.startswith("decoder_cnn.")  # Optional: Skip if not using contrastive
        ]

        encoder_state_dict = {k: pretrained_model_weights[k] for k in keys_to_load}
        supervised_model.load_state_dict(encoder_state_dict, strict=False)

        # Verify how many parameters were loaded
        num_loaded = len(encoder_state_dict)
        total_params = len(supervised_model.state_dict())
        print(f"Loaded {num_loaded}/{total_params} parameters. Missing keys are expected (e.g., classifier).")
        missing, unexpected = supervised_model.load_state_dict(encoder_state_dict, strict=False)
        print("Missing keys (expected):", missing)  # Should include classifier, pos_time, etc.
        print("Unexpected keys:", unexpected)  # Should be empty

        freeze_cnn_layers(supervised_model, num_frozen_layers=freeze_layer_cnn)  # Freeze first 3 CNN layers
        freeze_encoder_layers(supervised_model, num_frozen_layers=freeze_layer_encoder)  # Freeze first 3 transformer layers
        freeze_input_embeddings(supervised_model)  # Optional

        # Ensure classifier is ALWAYS trainable
        for param in supervised_model.classifier.parameters():
            param.requires_grad = True

        print_frozen_status(supervised_model)
    else:
        supervised_model = load_model_scratch(win_len, feature_size, depth, in_channels, num_classes,flag='joint')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Finetuning the model")
    criterion_task = nn.CrossEntropyLoss()
    # criterion_task = torch.nn.BCEWithLogitsLoss()
    supervised_support_loader, supervised_test_loader = load_acf_supervised(data_dir,task, BATCH_SIZE)
    print(f"Batch size in DataLoader: {supervised_support_loader.batch_size}")
    print("Data loaded")



    model, result = task_trainer_acf.supervised_train_acf(
                    model=supervised_model,
                    train_loader=supervised_support_loader,
                    valid_loader=supervised_test_loader,
                    num_epochs= num_epochs_2,
                    learning_rate=learning_rate,
                    criterion=criterion_task,
                    device=device,
                    save_path=results_dir,
                    optimizer_decay_rate=DECAY_RATE,
                    IF_TRANSFER=IF_TRANSFER,
                    weight_setting_string='none',
                    patience = patience,
                )
    result.to_csv(results_dir + 'results.csv')
    filename = os.path.join(results_dir, "best_model.pth.tar")
    best_model_state = copy.deepcopy(model.state_dict())
    torch.save(best_model_state, filename)

    fig_val_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=result['Epoch'], y=result['Val Accuracy'])
    plt.show()
    fig_val_acc.savefig(results_dir + "val_acc.png")

    fig_val_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=result['Epoch'], y=result['Val Loss'])
    plt.show()
    fig_val_loss.savefig(results_dir + "val_loss.png")

    fig_train_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=result['Epoch'], y=result['Train Accuracy'])
    plt.show()
    fig_train_acc.savefig(results_dir + "train_acc.png")

    fig_train_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=result['Epoch'], y=result['Train Loss'])
    plt.show()
    fig_train_loss.savefig(results_dir + "train_loss.png")

    data_dir_test = ".\\dataset\\task\\HM3\\CSIMAT100\\train\\test"
    unseen_test_loader = load_acf_unseen_environ(data_dir_test,task)
    test_result,if_save_results = task_trainer_acf.supervised_test_acf(
        model=model,
        tensor_loader=unseen_test_loader,
        criterion=criterion_task,
        device=device,
        num_classes=num_classes,
        task=task,
        save_path=results_dir
    )
    test_result.to_csv(results_dir + 'test_results.csv')
def run_task_acf_test():


    depth = 6
    in_channels = 1
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = ".\\experiments\\02132025\\task\\OWHM3\\CSI100_test\\best_model.pth.tar"
    results_dir = ".\\experiments\\02132025\\task\\OWHM3\\test\\"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    task = 'ThreeClass'
    data_dir = ".\\dataset\\task\\HM3\\CSIMAT100\\train\\test"
    criterion_task = nn.CrossEntropyLoss()
    model = ViT_MultiTask(
        emb_dim=128,
        encoder_heads=4,
        encoder_layers=6,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        # max_dist=32,  # for relative time
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0.1,
        num_classes=num_classes,
        c_out=60,
        freq_out=6,  # The final freq dimension we want in the CNN
        max_len=512  # or some large number for variable T
        )
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    print(f"Pretrained model loaded")
    unseen_test_loader = load_acf_unseen_environ(data_dir, task)
    test_result,if_save_results = task_trainer_acf.supervised_test_acf(
        model=model,
        tensor_loader=unseen_test_loader,
        criterion=criterion_task,
        device=device,
        num_classes=num_classes,
        task=task,
        save_path=results_dir
    )
    test_result.to_csv(results_dir + 'test_results.csv')
    return

def save_intermediates():
    BATCH_SIZE = 16
    task = "OWHM3"
    sample_rate = 100
    time_seg = 5
    win_len = sample_rate * time_seg
    supervised_support_loader, supervised_test_loader = save_data_supervised(task, BATCH_SIZE, win_len, sample_rate)
    return



if __name__ == "__main__":
    # save_intermediates()
    # run_task_csi()
    run_task_acf()
    # run_task_acf_test()