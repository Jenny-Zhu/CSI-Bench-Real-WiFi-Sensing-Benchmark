import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
from engine import ssl_trainer
from load import load_data_unsupervised, load_model_unsupervised
from tools.loss_fuction import NtXentLoss
from data.data_augmentation import DataAugmentation

import load

print("Attributes in load module:", dir(load))

# BATCH_SIZE = 8
# DECAY_RATE = 0.001
# learning_rate = 1e-4
# sample_rate = 100
# time_seg = 5
# win_len = sample_rate * time_seg
# feature_size = 56
# emb_size = int((win_len / 50) * (feature_size / 2))
# patience = 20
# num_epochs_1 = 100
# num_epochs_2 = 300
# depth = 3
# in_channels = 2
# model_name = "ViT"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# criterion_ssl = NtXentLoss()
# criterion_mc = nn.CrossEntropyLoss()
# augmentor = DataAugmentation(device=device)
# data_dir = "E:\Dataset\Dataset_OW\DatasetHP"
# results_dir = ".\\experiments\\test\\ssl\\"
# IoT_device = "ServerHP"
# number_of_classes = 4;
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--decay-rate', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--sample-rate', type=int, default=100)
    parser.add_argument('--time-seg', type=int, default=5)
    parser.add_argument('--feature-size', type=int, default=56)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--model-name', type=str, default='ViT')
    parser.add_argument('--classification-head-type', type=str, default='single_layer')
    parser.add_argument('--num-epochs-ssl', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--iot-device', type=str, default='ServerHP')
    return parser.parse_args()




def run_ssl(args):
    win_len = args.sample_rate * args.time_seg
    emb_size = int((win_len / 50) * (args.feature_size / 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_ssl = NtXentLoss()
    augmentor = DataAugmentation(device=device)

    unsupervised_train_loader = load_data_unsupervised(
        args.data_dir, args.iot_device, args.batch_size, win_len, args.sample_rate
    )

    model = load_model_unsupervised(
        win_len, args.feature_size, args.classification_head_type, depth=args.depth, in_channels=args.in_channels
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model, df1 = ssl_trainer.unsupervised_train(
        model, unsupervised_train_loader, args.num_epochs_ssl, args.learning_rate,
        criterion_ssl, device, augmentor, args.patience, args.output_dir
    )

    for loss_type in ['total_loss', 'kl_loss', 'he_loss', 'eh_loss', 'kde_loss']:
        fig = plt.figure(figsize=(7, 7))
        sn.lineplot(x=df1['Epochs'], y=df1[loss_type])
        plt.title(f'{loss_type} over Epochs')
        plt.savefig(os.path.join(args.output_dir, f'{loss_type}.png'))
        plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    run_ssl(args)
