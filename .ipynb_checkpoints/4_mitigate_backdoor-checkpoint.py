import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision import models
import random
import argparse
import numpy as np

# 从我们的共享工具箱和数据加载器中导入
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM, evaluate_clean_acc

# --- 评估函数 (包含之前修复的小bug) ---
def inject_trigger_video(inputs, trigger, trigger_size, position):
    B, C, T, H, W = inputs.shape; tr_h, tr_w = trigger_size; pos_x, pos_y = position
    trigger_T = trigger.unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1, 1)
    mask = torch.zeros_like(inputs); mask[:, :, :, pos_y:pos_y+tr_h, pos_x:pos_x+tr_w] = 1
    trigger_map = torch.zeros_like(inputs); trigger_map[:, :, :, pos_y:pos_y+tr_h, pos_x:pos_x+tr_w] = trigger_T
    # 修复了笔误，之前这里是 trigger_map * trigger
    return inputs * (1 - mask) + trigger_map

def evaluate_asr(model, dataloader, trigger, target_class_id, device):
    trigger_size = (trigger.shape[1], trigger.shape[2]); video_height, video_width = 224, 224
    position = (video_width - trigger_size[1], video_height - trigger_size[0])
    trigger_gpu = trigger.to(device); model.eval(); successful_attacks, total_non_target = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            non_target_mask = (labels != target_class_id)
            if non_target_mask.sum() == 0: continue
            inputs_to_attack = inputs[non_target_mask]; total_non_target += inputs_to_attack.size(0)
            inputs_with_trigger = inject_trigger_video(inputs_to_attack, trigger_gpu, trigger_size, position)
            outputs = model(inputs_with_trigger); _, predicted = torch.max(outputs.data, 1)
            successful_attacks += (predicted == target_class_id).sum().item()
    return 100 * successful_attacks / total_non_target if total_non_target > 0 else 0

# --- 主逻辑 ---
def main(args):
    device = f"cuda:{args.gpuid}"; torch.cuda.set_device(device)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    os.makedirs(args.output_dir, exist_ok=True)
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])
    full_train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"), transform=transform, num_frames=16)
    val_size = int(len(full_train_dataset) * args.val_ratio)
    _, clean_val_dataset = random_split(full_train_dataset, [len(full_train_dataset) - val_size, val_size])
    unlearning_loader = DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Using {len(clean_val_dataset)} samples for active unlearning.")

    clean_test_loader = DataLoader(UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/testlist01.txt"), transform=transform, num_frames=16), batch_size=args.batch_size, shuffle=False, num_workers=4)
    trigger = torch.load(args.trigger_path, map_location='cpu')
    trigger_gpu = trigger.to(device)
    trigger_size = (trigger.shape[1], trigger.shape[2]); video_height, video_width = 224, 224
    position = (video_width - trigger_size[1], video_height - trigger_size[0])

    net = CNN_LSTM(num_classes=10)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)

    # *** 核心修改 1: 精准选择需要优化的参数 ***
    print("Preparing for TARGETED unlearning: fine-tuning cnn.layer4, lstm, and fc layers.")
    params_to_optimize = []
    for name, param in net.named_parameters():
        if 'cnn.layer4' in name or 'lstm' in name or 'fc' in name:
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False
            
    optimizer = optim.Adam(params_to_optimize, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print("\n--- Evaluating model BEFORE mitigation ---")
    initial_acc = evaluate_clean_acc(net, clean_test_loader, device)
    initial_asr = evaluate_asr(net, clean_test_loader, trigger, args.target_label, device)
    print(f"Initial Clean ACC: {initial_acc:.2f}% | Initial ASR: {initial_asr:.2f}%")
    print("-" * 30)

    print("==> Starting TARGETED ACTIVE UNLEARNING defense...")
    for epoch in range(args.nb_epochs):
        # *** 核心修改 2: 精细化设置模型的 train/eval 模式 ***
        net.train() # 整体设为训练模式
        # 但将被冻结的部分明确设置为评估模式
        net.cnn.conv1.eval(); net.cnn.bn1.eval(); net.cnn.layer1.eval()
        net.cnn.layer2.eval(); net.cnn.layer3.eval()

        running_loss = 0.0
        for images, labels in unlearning_loader:
            images, labels = images.to(device), labels.to(device)
            antidote_images = inject_trigger_video(images, trigger_gpu, trigger_size, position)
            antidote_labels = labels
            
            optimizer.zero_grad()
            outputs = net(antidote_images)
            loss = criterion(outputs, antidote_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == args.nb_epochs - 1:
            acc = evaluate_clean_acc(net, clean_test_loader, device)
            asr = evaluate_asr(net, clean_test_loader, trigger, args.target_label, device)
            print(f"Epoch {epoch+1}/{args.nb_epochs} | Loss: {running_loss/len(unlearning_loader):.4f} | "
                  f"Clean ACC: {acc:.2f}% | ASR: {asr:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
            torch.save(net.state_dict(), os.path.join(args.output_dir, f"unlearned_model_epoch_{epoch+1}.pth"))
            
    print("\n--- Mitigation finished! ---")
    final_model_path = os.path.join(args.output_dir, "cleansed_model_optimized.pth")
    torch.save(net.state_dict(), final_model_path)
    print(f"Cleansed model with optimized ACC saved to: {final_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized Unlearning Defense for Video Backdoors')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--trigger-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output-dir', type=str, default='save/unlearning_defense_optimized/')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5) 
    parser.add_argument('--nb-epochs', type=int, default=12)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--target-label', type=int, default=0)
    parser.add_argument('--gpuid', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
    