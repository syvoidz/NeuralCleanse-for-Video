import os
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch import optim

# 从src目录导入所有核心组件
from src.ucf101_dataset import UCF101Dataset, PoisonedUCF101Dataset
from src.utils import CNN_LSTM, evaluate_clean_acc, evaluate_asr
from src.reconstructor import TriggerReconstructor
from src.defenses import unlearning_defense


def attack_and_defense_pipeline(args):
    """
    执行一个完整的、端到端的视频后门攻击与防御流水线。
    """
    # --- 0. 初始化 ---
    device = torch.device(args.device)
    torch.manual_seed(42);
    np.random.seed(42);
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 准备数据和干净模型 ---
    print("--- STEP 1: Loading Data and Clean Model ---")
    clean_model = CNN_LSTM(num_classes=args.num_classes)
    clean_model.load_state_dict(torch.load(args.clean_model_path, map_location='cpu', weights_only=True))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])

    train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                  split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"),
                                  transform=transform, num_frames=16)

    test_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                 split_file=os.path.join(args.data_dir, "splits/testlist01.txt"),
                                 transform=transform, num_frames=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # =======================================================
    #                       ATTACK PHASE
    # =======================================================
    print("\n--- STEP 2: Generating Trigger and Injecting Backdoor ---")

    reconstructor = TriggerReconstructor(clean_model, (3, 16, 224, 224), args.num_classes, args)
    trigger_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    mask, pattern = reconstructor.reconstruct(trigger_loader, target_class=args.target_label)
    trigger_data = {'mask': mask, 'pattern': pattern}

    poisoned_dataset = PoisonedUCF101Dataset(train_dataset, trigger_data, args.target_label, args.poison_rate)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    backdoor_model = CNN_LSTM(num_classes=args.num_classes).to(device)
    backdoor_model.load_state_dict(torch.load(args.clean_model_path, map_location=device, weights_only=True))

    optimizer_inject = optim.Adam(backdoor_model.parameters(), lr=args.lr_inject)
    criterion = torch.nn.CrossEntropyLoss()

    for name, param in backdoor_model.named_parameters():
        if 'cnn.layer4' in name or 'lstm' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for epoch in range(args.nb_epochs_inject):
        backdoor_model.train()
        backdoor_model.cnn.conv1.eval();
        backdoor_model.cnn.bn1.eval();
        backdoor_model.cnn.layer1.eval()
        backdoor_model.cnn.layer2.eval();
        backdoor_model.cnn.layer3.eval()
        for inputs, labels in poisoned_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_inject.zero_grad()
            outputs = backdoor_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_inject.step()

    print("Backdoor injection finished.")
    initial_acc = evaluate_clean_acc(backdoor_model, test_loader, device)
    initial_asr = evaluate_asr(backdoor_model, test_loader, trigger_data, args.target_label, device)
    print(f"Result of Attack -> Initial ACC: {initial_acc:.2f}%, Initial ASR: {initial_asr:.2f}%")

    # =======================================================
    #                       DEFENSE PHASE
    # =======================================================
    print("\n--- STEP 3: Mitigating Backdoor via Unlearning ---")

    val_size = int(len(train_dataset) * args.val_ratio)
    _, unlearning_dataset = random_split(train_dataset, [len(train_dataset) - val_size, val_size])
    unlearning_loader = DataLoader(unlearning_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    cleansed_model = unlearning_defense(backdoor_model, unlearning_loader, test_loader, trigger_data, args)

    final_acc = evaluate_clean_acc(cleansed_model, test_loader, device)
    final_asr = evaluate_asr(cleansed_model, test_loader, trigger_data, args.target_label, device)

    print("\n--- PIPELINE FINISHED ---")
    print(f"Final Results -> Clean ACC: {final_acc:.2f}%, ASR: {final_asr:.2f}%")

    defense_success_rate = (initial_asr - final_asr) / initial_asr if initial_asr > 1e-6 else 0.0

    return {
        'initial_clean_acc': initial_acc,
        'initial_asr': initial_asr,
        'final_clean_acc': final_acc,
        'final_asr': final_asr,
        'defense_success_rate_percent': defense_success_rate * 100
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a full Attack-Defense Pipeline for Video Backdoors')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--clean_model_path', type=str, default='./models/cnnlstm-ucf10_benign_best.pth')
    parser.add_argument('--output_dir', type=str, default='run_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--lr_reconstruct', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--init_cost', type=float, default=1e-3)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--cost_multiplier', type=float, default=2.0)
    parser.add_argument('--lr_inject', type=float, default=5e-5)
    parser.add_argument('--nb_epochs_inject', type=int, default=20)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--lr_mitigate', type=float, default=2e-5)
    parser.add_argument('--nb_epochs_mitigate', type=int, default=30)
    parser.add_argument('--val_ratio', type=float, default=0.05)

    args = parser.parse_args()

    final_results = attack_and_defense_pipeline(args)

    print("\n=================================")
    print("           FINAL REPORT          ")
    print("=================================")
    print(f"Initial Clean Accuracy:       {final_results['initial_clean_acc']:.2f}%")
    print(f"Initial Attack Success Rate:  {final_results['initial_asr']:.2f}%")
    print("-" * 33)
    print(f"Final Clean Accuracy:         {final_results['final_clean_acc']:.2f}%")
    print(f"Final Attack Success Rate:    {final_results['final_asr']:.2f}%")
    print("=================================")
    print(f"Defense Success Rate:         {final_results['defense_success_rate_percent']:.2f}%")
    print("=================================")
