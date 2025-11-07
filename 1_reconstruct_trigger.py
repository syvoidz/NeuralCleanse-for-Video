import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision import models
import random
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# 假设 ucf101_dataset.py 和 utils.py 都在 ./src/ 目录下
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM

# --- 辅助函数 ---
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

class TriggerReconstructor:
    def __init__(self, model, input_shape, num_classes, args):
        self.model = model.to(args.device)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.args = args
        self.device = args.device

        self.mask_tanh = nn.Parameter(torch.zeros((1, input_shape[2], input_shape[3]), device=self.device))
        self.pattern_tanh = nn.Parameter(torch.zeros((input_shape[0], input_shape[2], input_shape[3]), device=self.device))
        
        self.optimizer = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=args.lr)

    def get_trigger(self):
        mask = (torch.tanh(self.mask_tanh) + 1) / 2 # -> (0, 1)
        pattern = (torch.tanh(self.pattern_tanh) + 1) / 2 # -> (0, 1)
        return mask, pattern

    def apply_trigger(self, inputs):
        mask, pattern = self.get_trigger()
        mask = mask.to(inputs.device)
        pattern = pattern.to(inputs.device)
        
        pattern_time = pattern.unsqueeze(1).repeat(1, self.input_shape[1], 1, 1)
        triggered_inputs = (1 - mask) * inputs + mask * pattern_time
        return triggered_inputs

    def reconstruct(self, dataloader, target_class):
        nn.init.uniform_(self.mask_tanh, a=-3.0, b=3.0)
        nn.init.uniform_(self.pattern_tanh, a=-3.0, b=3.0)
        lambda_val = self.args.init_cost
        cost_up_counter, cost_down_counter = 0, 0
        best_mask_norm = float('inf')
        best_mask, best_pattern = None, None

        Y_target_batch = torch.full((self.args.batch_size,), target_class, dtype=torch.long, device=self.device)
        iterator = iter(dataloader)

        for step in tqdm(range(self.args.steps), desc=f"Reconstructing for class {target_class}"):
            try:
                inputs, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, _ = next(iterator)
            inputs = inputs.to(self.device)
            Y_target = Y_target_batch[:inputs.shape[0]]

            self.model.train()
            freeze_bn(self.model)

            triggered_inputs = self.apply_trigger(inputs)
            outputs = self.model(triggered_inputs)
            
            loss_ce = nn.CrossEntropyLoss()(outputs, Y_target)
            mask, _ = self.get_trigger()
            loss_reg = torch.sum(torch.abs(mask))
            total_loss = loss_ce + lambda_val * loss_reg
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                outputs_eval = self.model(triggered_inputs)
                attack_succ_rate = (torch.argmax(outputs_eval, dim=1) == Y_target).float().mean().item()

            if attack_succ_rate >= self.args.attack_succ_threshold:
                cost_up_counter += 1; cost_down_counter = 0
            else:
                cost_up_counter = 0; cost_down_counter += 1

            if cost_up_counter >= self.args.patience:
                cost_up_counter = 0; lambda_val *= self.args.cost_multiplier
            elif cost_down_counter >= self.args.patience:
                cost_down_counter = 0; lambda_val /= self.args.cost_multiplier

            current_mask_norm = loss_reg.item()
            if attack_succ_rate >= self.args.attack_succ_threshold and current_mask_norm < best_mask_norm:
                best_mask_norm = current_mask_norm
                best_mask, best_pattern = self.get_trigger()
                tqdm.write(f"Step {step}: New best trigger found! Mask L1 norm: {best_mask_norm:.2f}")

        if best_mask is None:
            print("Warning: Could not find an effective trigger for this class.")
            best_mask, best_pattern = self.get_trigger()
        
        return best_mask.detach().cpu(), best_pattern.detach().cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Step 1: Trigger Reconstruction')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--clean_model_path', type=str, default='./models/cnnlstm-ucf10_benign_best.pth')
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--init_cost', type=float, default=1e-3)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--cost_multiplier', type=float, default=1.5)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = CNN_LSTM(num_classes=10)
    model.load_state_dict(torch.load(args.clean_model_path, map_location='cpu'))
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])
    dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"), transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    reconstructor = TriggerReconstructor(model, input_shape=(3, 16, 224, 224), num_classes=10, args=args)
    
    target_class_for_attack = 0
    
    print(f"\n--- Starting trigger reconstruction for target class {target_class_for_attack} ---")
    mask, pattern = reconstructor.reconstruct(dataloader, target_class=target_class_for_attack)
    
    # --- 改进的输出形式 ---
    # 1. 保存可编程的触发器对象
    trigger_data = {'mask': mask, 'pattern': pattern}
    save_path_pth = os.path.join(args.output_dir, f'trigger_target_{target_class_for_attack}.pth')
    torch.save(trigger_data, save_path_pth)
    print(f"--- Optimized trigger object saved to {save_path_pth} ---")

    # 2. 保存直观的可视化图片
    # 创建一个只包含触发器的图像 (mask * pattern)
    trigger_visual = mask * pattern
    save_path_png = os.path.join(args.output_dir, f'trigger_target_{target_class_for_attack}.png')
    save_image(trigger_visual, save_path_png)
    print(f"--- Trigger visualization saved to {save_path_png} ---")
    