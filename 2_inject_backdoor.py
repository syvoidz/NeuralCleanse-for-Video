import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import random
import argparse
import numpy as np
from tqdm import tqdm

# 从我们的共享工具箱中导入核心组件
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM, evaluate_clean_acc, evaluate_asr

# --- 辅助函数：Poisoned Dataset ---
# 注意：这里的 inject_trigger_video 与 utils.py 中的不同，因为它直接使用 mask 和 pattern
def inject_trigger_with_mask_pattern(inputs, mask, pattern):
    # inputs: (B, C, T, H, W)
    # mask: (1, H, W)
    # pattern: (C, H, W)
    
    mask = mask.to(inputs.device)
    pattern = pattern.to(inputs.device)
    
    # 将 mask 扩展为 (B, C, T, H, W)
    # 使用 view 来创建正确的维度，然后用 expand 高效地“重复”
    # view(1, 1, 1, H, W) -> 创建一个5D张量
    mask_expanded = mask.view(1, 1, 1, mask.shape[1], mask.shape[2]).expand_as(inputs)
    
    # 将 pattern 扩展为 (B, C, T, H, W)
    # view(1, C, 1, H, W) -> 创建一个5D张量
    pattern_expanded = pattern.view(1, pattern.shape[0], 1, pattern.shape[1], pattern.shape[2]).expand_as(inputs)
    
    return (1 - mask_expanded) * inputs + mask_expanded * pattern_expanded

class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, trigger_data, target_class_id, poison_rate):
        self.original_dataset = original_dataset
        self.mask = trigger_data['mask'].cpu()
        self.pattern = trigger_data['pattern'].cpu()
        self.target_class_id = target_class_id
        self.poison_rate = poison_rate
        
        dataset_size = len(self.original_dataset)
        num_poisoned = int(dataset_size * poison_rate)
        self.poisoned_indices = set(random.sample(range(dataset_size), num_poisoned))
        print(f"Poisoned dataset created: {num_poisoned}/{dataset_size} samples will be poisoned.")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        video, label = self.original_dataset[idx]
        if idx in self.poisoned_indices:
            # unsqueeze(0) 创建了一个 (1, C, T, H, W) 的伪批次，正好是新函数所期望的
            triggered_video = inject_trigger_with_mask_pattern(video.unsqueeze(0), self.mask, self.pattern).squeeze(0)
            return triggered_video, torch.tensor(self.target_class_id, dtype=torch.long)
        else:
            return video, label

# --- 主逻辑 ---
def main(args):
    device = args.device; torch.cuda.set_device(device)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    
    # --- 数据加载 ---
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])
    
    clean_train_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"), transform=transform, num_frames=16)
    
    print(f"Loading trigger from: {args.trigger_path}")
    trigger_data = torch.load(args.trigger_path, map_location='cpu')
    
    poisoned_train_dataset = PoisonedDataset(clean_train_dataset, trigger_data, args.target_label, args.poison_rate)
    poisoned_train_dataloader = DataLoader(poisoned_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    clean_test_dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/testlist01.txt"), transform=transform, num_frames=16)
    clean_test_dataloader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 模型与优化器 ---
    model = CNN_LSTM(num_classes=10)
    print(f"Loading clean model weights from: {args.clean_model_path}")
    model.load_state_dict(torch.load(args.clean_model_path, map_location=device))
    model.to(device)
    
    # 我们将微调整个模型以获得最强的后门效果
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # --- 微调循环 ---
    print("Starting fine-tuning to inject backdoor...")
    best_asr = 0.0
    for epoch in range(args.nb_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in tqdm(poisoned_train_dataloader, desc=f"Epoch {epoch+1}/{args.nb_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        # --- 评估 ---
        # 重新组合触发器用于评估
        eval_trigger = trigger_data['mask'] * trigger_data['pattern']
        acc = evaluate_clean_acc(model, clean_test_dataloader, device)
        asr = evaluate_asr(model, clean_test_dataloader, eval_trigger, args.target_label, device)
        
        print(f"Epoch {epoch+1}/{args.nb_epochs} | Loss: {running_loss/len(poisoned_train_dataloader):.4f} | "
              f"Clean ACC: {acc:.2f}% | ASR: {asr:.2f}%")

        if asr > best_asr and asr > 90.0 and acc > 75.0:
            best_asr = asr
            torch.save(model.state_dict(), args.output_model_path)
            print(f"*** New best backdoor model saved (ACC: {acc:.2f}%, ASR: {asr:.2f}%) ***")
            
    print(f"\n--- Backdoor model training finished ---\nBest model saved to {args.output_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Step 2: Backdoor Injection')
    
    # --- 路径参数 ---
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--clean_model_path', type=str, default='./models/cnnlstm-ucf10_benign_best.pth')
    parser.add_argument('--trigger_path', type=str, default='./results/trigger_target_0.pth', help='Path to the trigger object from step 1')
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output_model_path', type=str, default='./models/backdoor_model_nc.pth')
    
    # --- 训练和攻击参数 ---
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
    