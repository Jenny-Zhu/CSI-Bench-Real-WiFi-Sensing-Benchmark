import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import random
import numpy as np
# Import from your local modules
from engine import ssl_trainer_joint as ssl_trainer
from engine.meta_trainer import *
from load import (
    load_acf_data_unsupervised,
    load_csi_data_unsupervised,
    load_model_unsupervised_joint_csi_var,
    load_data_unsupervised,
    load_model_unsupervised_joint,
    load_csi_data_benchmark,
    load_csi_model_benchmark
)
from tools.loss_fuction import NtXentLoss
from data.data_augmentation import DataAugmentation, DataAugmentACF

def set_seed(seed=111):
    random.seed(seed)                  # Python random
    np.random.seed(seed)                # Numpy random
    torch.manual_seed(seed)             # Torch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    # Torch current GPU
        torch.cuda.manual_seed_all(seed)  # Torch all GPUs (multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()

    # Existing args
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--decay-rate', type=float, default=0.001)
    parser.add_argument('--num-epochs-1', type=int, default=100)
    parser.add_argument('--num-epochs-2', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sample-rate', type=int, default=100)
    parser.add_argument('--time-seg', type=int, default=5)
    parser.add_argument('--mode', type=str, default='csi')  # now mode can be 'csi' or 'maml_csi'
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--emb-size', type=int, default=128)
    parser.add_argument('--freq-out', type=int, default=10)

    # parser.add_argument('--csi-data-dir', type=str, default='/opt/ml/input/data/csi')
    parser.add_argument('--csi-data-dir', nargs='+', type=str, default=['/opt/ml/input/data/csi'])
    parser.add_argument('--acf-data-dir', type=str, default='/opt/ml/input/data/acf')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--model-name', type=str, default='WiT') # for benchmark: CNN
    parser.add_argument('--results-subdir', type=str, default='test_ssl')
    parser.add_argument('--iot-device', type=str, default='ServerHP')

    # Meta_learning args
    parser.add_argument('--task1-name', type=str, default='goodbad')  # goodbad or motionempty
    parser.add_argument('--task2-name', type=str, default='motionempty')  # goodbad or motionempty
    parser.add_argument('--meta-learning-method', type=str, default='maml')

    # MAML-specific args
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--tasks-per-batch', type=int, default=4)
    parser.add_argument('--inner-lr', type=float, default=0.01)
    parser.add_argument('--meta-lr', type=float, default=0.001)
    parser.add_argument('--k-shot', type=int, default=5)
    parser.add_argument('--q-query', type=int, default=15)
    parser.add_argument('--resize-height', type=int, default=64)
    parser.add_argument('--resize-width', type=int, default=100)


    return parser.parse_args()


def run_benchmark_csi(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_csi = args.csi_data_dir
    meta_learn_method = args.meta_learning_method
    meta_learn_method = 'maml'
    task1_name = args.task1_name
    task2_name = args.task2_name
    if task1_name == 'goodbad':
        data_dir_csi =  [
            r'D:\AnomalyCSIVerification_dataset\data\CSI_verification\data_for_benchmark'
        ] # the data folder list

    # Few-shot task config
    k_shot = args.k_shot
    q_query = args.q_query
    resize_height = args.resize_height
    resize_width = args.resize_width

    # Set the task name (e.g., goodbad or motionempty)
    if task1_name== 'goodbad':
        label_keywords = {'good': 0, 'bad': 1}
    elif task2_name == 'motionempty':
        label_keywords = {'empty': 0, 'motion': 1}
    else:
        raise ValueError("Invalid task name.")

    print(f"\n--- Training on task: {task1_name} ---")
    print(f"\n--- Training on task: {task2_name} ---")

    train_datasets, task_dataset = load_csi_data_benchmark(data_dir_csi, resize_height, resize_width, label_keywords, k_shot, q_query)
    # val_datasets, val_task_dataset = load_csi_data_benchmark(val_folder_paths, resize_height, resize_width, label_keywords, k_shot, q_query)
    
    # Infer (H, W) automatically
    x_s, _, _, _ = train_datasets[0].sample_task()
    _, _, H, W = x_s.shape

    ## load the model
    model = load_csi_model_benchmark(H, W, device)

    # Run MAML training
    if meta_learn_method == 'maml':
        maml_train(
            model=model,
            task_dataset=task_dataset,
            device = device,
            steps=args.steps,
            tasks_per_batch=args.tasks_per_batch,
            inner_lr=args.inner_lr,
            meta_lr=args.meta_lr
        )
    
    if meta_learn_method == 'lstm':
        meta_optimizer = LSTMOptimizer(hidden_size=20).to(device)

        lstm_meta_train(
            model=model,
            meta_optimizer=meta_optimizer,
            task_dataset=task_dataset,
            device=device,
            steps=args.steps,
            tasks_per_batch=args.tasks_per_batch
        )
    

    # save the model
    save_path = os.path.join(args.output_dir, args.results_subdir, f"{args.model_name}")
    os.makedirs(save_path, exist_ok=True)

def main():
    set_seed(123)  
    args = parse_args()

    if args.mode.lower() == 'csi':
        run_benchmark_csi(args)
    else:
        print(f"Unknown mode: {args.mode}. Please choose from 'csi', 'acf'.")


if __name__ == "__main__":
    main()