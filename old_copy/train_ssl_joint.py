import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
# Import from your local modules
from engine import ssl_trainer_joint as ssl_trainer
from load import (
    load_acf_data_unsupervised,
    load_csi_data_unsupervised,
    load_model_unsupervised_joint_csi_var,
    load_data_unsupervised,
    load_model_unsupervised_joint
)
from tools.loss_fuction import NtXentLoss
from data.data_augmentation import DataAugmentation, DataAugmentACF


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--decay-rate', type=float, default=0.001)
    parser.add_argument('--num-epochs-1', type=int, default=100)
    parser.add_argument('--num-epochs-2', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sample-rate', type=int, default=100)
    parser.add_argument('--time-seg', type=int, default=5)
    parser.add_argument('--mode', type=str, default='csi')
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--emb-size', type=int, default=128)
    parser.add_argument('--freq-out', type=int, default=10)

    parser.add_argument('--csi-data-dir', type=str, default='/opt/ml/input/data/csi')
    parser.add_argument('--acf-data-dir', type=str, default='/opt/ml/input/data/acf')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--model-name', type=str, default='WiT')
    parser.add_argument('--results-subdir', type=str, default='test_ssl')
    parser.add_argument('--iot-device', type=str, default='ServerHP')

    return parser.parse_args()


def run_ssl_csi(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_acf = args.csi_data_dir
    train_loader = load_csi_data_unsupervised(data_dir_acf, args.batch_size)

    model = load_model_unsupervised_joint_csi_var(
        emb_size=args.emb_size,
        depth=args.depth,
        freq_out=args.freq_out,
        in_channels=args.in_channels
    )

    save_path = os.path.join(args.output_dir, args.results_subdir, f"{args.model_name}")
    os.makedirs(save_path, exist_ok=True)

    model, df1 = ssl_trainer.unsupervised_train_acf(
        model,
        train_loader,
        args.num_epochs_1,
        args.learning_rate,
        NtXentLoss(),
        device,
        DataAugmentACF(),
        args.patience,
        save_path
    )
    print("CSI training done!")


def run_ssl_acf(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_acf = args.acf_data_dir
    batch_size = args.batch_size
    win_len = 250
    feature_size = 98

    model = load_model_unsupervised_joint(
        win_len,
        feature_size,
        depth=args.depth,
        in_channels=args.in_channels
    )

    save_path = os.path.join(args.output_dir, args.results_subdir, f"{args.model_name}")
    os.makedirs(save_path, exist_ok=True)

    train_loader = load_acf_data_unsupervised(data_dir_acf, batch_size)

    model, df1 = ssl_trainer.unsupervised_train_acf(
        model,
        train_loader,
        args.num_epochs_1,
        args.learning_rate,
        NtXentLoss(),
        device,
        DataAugmentACF(feature_size=feature_size),
        args.patience,
        save_path
    )
    print("ACF training done!")


def save_intermediates(args):
    print("save_intermediates() is not fully implemented. Add your logic here if needed.")


def main():
    args = parse_args()

    if args.mode.lower() == 'csi':
        run_ssl_csi(args)
    elif args.mode.lower() == 'acf':
        run_ssl_acf(args)
    else:
        print(f"Unknown mode: {args.mode}. Please choose from 'csi', 'acf'.")


if __name__ == "__main__":
    main()
