import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import random
import argparse
import numpy as np
from tqdm import tqdm

# 从我们的共享工具箱和数据加载器中导入所有核心组件
from src.ucf101_dataset import UCF101Dataset, PoisonedUCF101Dataset
from src.utils import CNN_LSTM, evaluate_clean_acc, evaluate_asr


def main(args):
    """
    主工作流：加载干净模型和触发器，使用被污染的数据集进行微调，
    评估后门效果，并保存最优的后门模型。
    """
    # --- 1. 初始化和环境设置 ---
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)  # 保证实验的可复现性

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. 数据加载 ---
    print("==> Preparing data...")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])

    # 加载干净的训练集，作为投毒的基础
    clean_train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                        split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"),
                                        transform=transform, num_frames=16)

    print(f"Loading trigger from: {args.trigger_path}")
    trigger_data = torch.load(args.trigger_path, map_location='cpu')

    # 使用PoisonedUCF101Dataset来封装和污染数据
    poisoned_train_dataset = PoisonedUCF101Dataset(clean_train_dataset, trigger_data, args.target_label,
                                                   args.poison_rate)
    poisoned_train_dataloader = DataLoader(poisoned_train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=4)

    # 加载干净的测试集，用于评估
    clean_test_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                       split_file=os.path.join(args.data_dir, "splits/testlist01.txt"),
                                       transform=transform, num_frames=16)
    clean_test_dataloader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 3. 模型与优化器 ---
    model = CNN_LSTM(num_classes=args.num_classes).to(device)
    print(f"Loading clean model weights from: {args.clean_model_path}")
    model.load_state_dict(torch.load(args.clean_model_path, map_location=device))

    # 我们将微调整个模型以获得最强的后门效果
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # --- 4. 微调循环 ---
    print("==> Starting fine-tuning to inject backdoor...")
    best_asr = 0.0
    for epoch in range(args.nb_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(poisoned_train_dataloader, desc=f"Epoch {epoch + 1}/{args.nb_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.4f}")

        scheduler.step()

        # --- 5. 评估 ---
        # 组合 mask 和 pattern 来创建用于评估的触发器
        eval_trigger = trigger_data['mask'] * trigger_data['pattern']
        acc = evaluate_clean_acc(model, clean_test_dataloader, device)
        asr = evaluate_asr(model, clean_test_dataloader, eval_trigger, args.target_label, device)

        print(f"Epoch {epoch + 1}/{args.nb_epochs} Results | Clean ACC: {acc:.2f}% | ASR: {asr:.2f}%")

        # 保存ASR最高且ACC不太低的模型
        if asr > best_asr and asr > 90.0 and acc > 75.0:
            best_asr = asr
            torch.save(model.state_dict(), args.output_model_path)
            print(f"*** New best backdoor model saved (ACC: {acc:.2f}%, ASR: {asr:.2f}%) ***")

    print(f"\n--- Backdoor model training finished ---\nBest model saved to {args.output_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Step 2: Backdoor Injection')

    # --- 路径参数 ---
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--clean_model_path', type=str, default='./models/cnnlstm-ucf10_benign_best.pth',
                        help='Path to the clean base model')
    parser.add_argument('--trigger_path', type=str, default='./results/trigger_target_0.pth',
                        help='Path to the trigger object from step 1')
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled', help='Path to the dataset directory')
    parser.add_argument('--output_model_path', type=str, default='./models/backdoor_model_nc.pth',
                        help='Path to save the final backdoored model')

    # --- 训练和攻击参数 ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--nb_epochs', type=int, default=20, help='Number of epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--poison_rate', type=float, default=0.1, help='Proportion of training data to poison')
    parser.add_argument('--target_label', type=int, default=0, help='The target class for the backdoor attack')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset. Set to 101 for full UCF-101.')

    args = parser.parse_args()
    main(args)
