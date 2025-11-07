import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import argparse
import numpy as np

# 从src目录导入所有核心组件
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM, evaluate_clean_acc, evaluate_asr
from src.reconstructor import TriggerReconstructor
from src.defenses import unlearning_defense


def outlier_detection(l1_norm_list):
    """从 3_detect_backdoor.py 迁移过来的 MAD 检测函数"""
    consistency_constant = 1.4826
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    if mad < 1e-9: mad = 1e-9  # 避免除以0

    min_mad_score = np.abs(np.min(l1_norm_list) - median) / mad

    print("\n--- [PHASE 1] Anomaly Detection Results ---")
    print(f"Median L1 Norm: {median:.2f}, MAD: {mad:.2f}")
    print(f"Anomaly Index (of min L1 norm): {min_mad_score:.2f}")

    flagged_labels = [i for i, norm in enumerate(l1_norm_list) if (np.abs(norm - median) / mad > 2 and norm < median)]

    if len(flagged_labels) > 0:
        print(f"CONCLUSION: Backdoor DETECTED. Flagged Label(s): {flagged_labels}")
    else:
        print("CONCLUSION: No backdoor detected based on the anomaly threshold.")

    return min_mad_score, flagged_labels


def main(args):
    """主工作流：可选的自动检测 + 靶向缓解"""
    device = torch.device(args.device)
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 加载模型和数据 ---
    model = CNN_LSTM(num_classes=args.num_classes)
    print(f"Loading suspicious model from: {args.backdoor_model_path}")
    model.load_state_dict(torch.load(args.backdoor_model_path, map_location='cpu'))
    model.to(device)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])

    # --- 2. 自动检测阶段 (如果需要) ---
    if not args.skip_detection:
        print("\n--- Starting [PHASE 1]: Automatic Detection ---")
        # 为检测准备数据加载器
        full_train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                           split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"),
                                           transform=transform, num_frames=16)
        detection_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        reconstructor = TriggerReconstructor(model, (3, 16, 224, 224), args.num_classes, args)
        l1_norm_list = []
        reconstructed_triggers = {}
        for class_idx in range(args.num_classes):
            mask, pattern = reconstructor.reconstruct(detection_loader, target_class=class_idx)
            reconstructed_triggers[class_idx] = {'mask': mask, 'pattern': pattern}
            l1_norm_list.append(torch.sum(torch.abs(mask)).item())

        _, flagged_labels = outlier_detection(np.array(l1_norm_list))

        if not flagged_labels:
            print("\nModel appears to be clean. Mitigation is not required. Exiting.")
            return  # 退出程序

        # 自动设置后门标签和触发器路径
        args.target_label = flagged_labels[0]
        trigger_data = reconstructed_triggers[args.target_label]
        print(f"Mitigation will target the automatically detected label: {args.target_label}")
    else:
        print("\n--- [PHASE 1] Skipped. Using provided target label and trigger path. ---")
        trigger_data = torch.load(args.trigger_path, map_location='cpu')

    # --- 3. 缓解阶段 ---
    # 为Unlearning准备干净验证集
    full_train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                       split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"),
                                       transform=transform, num_frames=16)
    val_size = int(len(full_train_dataset) * args.val_ratio)
    _, clean_val_dataset = random_split(full_train_dataset, [len(full_train_dataset) - val_size, val_size])
    unlearning_loader = DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 为评估准备测试集
    test_loader = DataLoader(UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                                           split_file=os.path.join(args.data_dir, "splits/testlist01.txt"),
                                           transform=transform, num_frames=16), batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # 评估修复前的模型
    print("\n--- Evaluating model BEFORE mitigation ---")
    initial_acc = evaluate_clean_acc(model, test_loader, device)
    initial_asr_trigger = trigger_data['mask'] * trigger_data['pattern']
    initial_asr = evaluate_asr(model, test_loader, initial_asr_trigger, args.target_label, device)
    print(f"Initial Clean ACC: {initial_acc:.2f}% | Initial ASR: {initial_asr:.2f}%")

    # 调用防御引擎
    cleansed_model = unlearning_defense(model, unlearning_loader, test_loader, trigger_data, args)

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "cleansed_model.pth")
    torch.save(cleansed_model.state_dict(), final_model_path)
    print(f"Cleansed model saved to: {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Full Mitigation Pipeline')

    # --- 核心参数 ---
    parser.add_argument('--backdoor_model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=10)

    # --- 任务控制 ---
    parser.add_argument('--skip_detection', action='store_true',
                        help='Skip detection and directly mitigate using --trigger_path and --target_label')

    # --- 路径 ---
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output_dir', type=str, default='results_mitigation/')
    parser.add_argument('--trigger_path', type=str, default='./results/trigger_target_0.pth',
                        help='Path to trigger if --skip_detection is used')

    # --- 检测超参数 ---
    parser.add_argument('--lr_reconstruct', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--init_cost', type=float, default=1e-3)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--cost_multiplier', type=float, default=2.0)

    # --- 缓解超参数 ---
    parser.add_argument('--lr_mitigate', type=float, default=1e-5)
    parser.add_argument('--nb_epochs_mitigate', type=int, default=12, dest='nb_epochs')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--target_label', type=int, default=0,
                        help='The target class label (used if --skip_detection is on)')

    # --- 通用 ---
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    main(args)
