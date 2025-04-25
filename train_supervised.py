import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from tqdm import tqdm

# 导入训练引擎
from engine.supervised.task_trainer import TaskTrainer
from engine.supervised.task_trainer_acf import TaskTrainerACF

# 导入数据加载器
from load import (
    load_csi_supervised_integrated,
    load_csi_unseen_integrated,
    load_acf_supervised,
    load_acf_unseen_environ,
    load_model_scratch
)

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='训练监督学习模型')
    
    # 数据目录
    parser.add_argument('--csi-data-dir', type=str, default=None,
                      help='包含CSI数据的目录')
    parser.add_argument('--acf-data-dir', type=str, default=None,
                      help='包含ACF数据的目录')
    parser.add_argument('--output-dir', type=str, default='experiments',
                      help='保存输出结果的目录')
    parser.add_argument('--results-subdir', type=str, default='supervised',
                      help='输出目录下保存结果的子目录')
    
    # 数据参数
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'], default='csi',
                      help='使用的数据模态(csi或acf)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='用于训练的数据比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='用于验证的数据比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                      help='用于测试的数据比例')
    parser.add_argument('--win-len', type=int, default=250,
                      help='CSI数据的窗口长度')
    parser.add_argument('--feature-size', type=int, default=90,
                      help='CSI数据的特征大小')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num-epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=15, help='早停的等待轮数')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=2, help='类别数量')
    parser.add_argument('--in-channels', type=int, default=1, help='输入通道数')
    parser.add_argument('--freeze-backbone', action='store_true', help='是否冻结骨干网络')
    
    # 数据参数
    parser.add_argument('--unseen-test', action='store_true', help='是否在未见过的环境上测试')
    parser.add_argument('--integrated-loader', action='store_true', help='是否使用集成数据加载器')
    parser.add_argument('--task', type=str, default='ThreeClass', 
                      help='集成加载器的任务类型(如ThreeClass, HumanNonhuman)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=None, help='训练设备')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以便结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_supervised_csi(args):
    """训练CSI模态的模型"""
    print(f"开始CSI模态的监督学习训练...")

    # 加载数据
    train_loader, val_loader, test_loader = load_csi_supervised_integrated(
        args.csi_data_dir,
        task=args.task,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print(f"已加载CSI数据，任务: {args.task}")

    # 加载模型
    model = load_model_scratch(
        num_classes=args.num_classes,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=args.in_channels
    )
    print("使用随机初始化的模型")
    
    # 设置保存路径
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                            f"{args.task}_{args.model_name}_csi")
    os.makedirs(save_path, exist_ok=True)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器并开始训练
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, 
        criterion=criterion,
        device=args.device,
        save_path=save_path,
        optimizer_decay_rate=args.weight_decay,
        patience=args.patience
    )
    
    model, results_df = trainer.train()
    
    # 保存训练结果
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # 测试模型
    print("在测试集上评估模型...")
    test_results, _ = trainer.test(test_loader, args.num_classes, args.task)
    test_results.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
    
    # 绘制结果
    plot_results(results_df, save_path)
    
    print(f"CSI监督学习训练完成！模型和结果保存到 {save_path}")
    
    return model, results_df


def train_supervised_acf(args):
    """训练ACF模态的模型"""
    print(f"开始ACF模态的监督学习训练...")
    
    # 准备数据
    if args.unseen_test:
        # 对于未见过的测试环境
        test_loader = load_acf_unseen_environ(
            args.acf_data_dir,
            task=args.task
        )
        print("使用带有未见过环境的测试集...")
        
        # 从另一个目录加载训练和验证数据
        if hasattr(args, 'train_data_dir') and args.train_data_dir:
            train_dir = args.train_data_dir
        else:
            # 尝试使用父目录
            train_dir = os.path.dirname(args.acf_data_dir.rstrip('/\\'))
            if not train_dir or train_dir == args.acf_data_dir:
                train_dir = args.acf_data_dir
        
        # 加载训练数据
        print(f"从以下位置加载训练数据: {train_dir}")
        train_loader, val_loader, _ = load_acf_supervised(
            train_dir,
            task=args.task,
            batch_size=args.batch_size
        )
    else:
        # 正常训练模式
        train_loader, val_loader, test_loader = load_acf_supervised(
            args.acf_data_dir,
            task=args.task,
            batch_size=args.batch_size
        )
        print(f"使用ACF数据加载器，任务: {args.task}")
    
    # 加载模型
    model = load_model_scratch(
        num_classes=args.num_classes,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=args.in_channels
    )
    print("使用随机初始化的模型")
    
    # 设置保存路径
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                           f"{args.task}_{args.model_name}_acf")
    os.makedirs(save_path, exist_ok=True)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器并开始训练
    trainer = TaskTrainerACF(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        criterion=criterion,
        device=args.device,
        save_path=save_path,
        optimizer_decay_rate=args.weight_decay,
        IF_TRANSFER=False,
        weight_setting_string='none',
        patience=args.patience
    )
    
    model, results_df = trainer.train()
    
    # 保存训练结果
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # 测试模型
    print("在测试集上评估模型...")
    test_results, _ = trainer.test(test_loader, args.num_classes, args.task, save_path)
    test_results.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
    
    # 绘制结果
    plot_results(results_df, save_path)
    
    print(f"ACF监督学习训练完成！模型和结果保存到 {save_path}")
    
    return model, results_df


def plot_results(results, save_path):
    """绘制训练和验证结果曲线"""
    # 绘制验证准确率曲线
    fig_val_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Val Accuracy'])
    plt.title('验证准确率')
    plt.savefig(os.path.join(save_path, "val_acc.png"))
    plt.close()
    
    # 绘制验证损失曲线
    fig_val_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Val Loss'])
    plt.title('验证损失')
    plt.savefig(os.path.join(save_path, "val_loss.png"))
    plt.close()
    
    # 绘制训练准确率曲线
    fig_train_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Train Accuracy'])
    plt.title('训练准确率')
    plt.savefig(os.path.join(save_path, "train_acc.png"))
    plt.close()
    
    # 绘制训练损失曲线
    fig_train_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Train Loss'])
    plt.title('训练损失')
    plt.savefig(os.path.join(save_path, "train_loss.png"))
    plt.close()


def main(args=None):
    """主函数"""
    # 如果没有提供参数，则解析命令行参数
    if args is None:
        args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 根据数据模态选择训练函数
    if args.mode.lower() == 'csi':
        train_supervised_csi(args)
    elif args.mode.lower() == 'acf':
        train_supervised_acf(args)
    else:
        raise ValueError(f"未知的模态: {args.mode}。请选择'csi'或'acf'")


if __name__ == "__main__":
    main() 