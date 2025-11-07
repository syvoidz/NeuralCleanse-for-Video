import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# 导入我们的核心模块
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM

# --- 我们需要从 1_reconstruct_trigger.py 中复制 TriggerReconstructor 类过来 ---
# (为了让这个脚本可以独立运行)
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

class TriggerReconstructor:
    def __init__(self, model, input_shape, num_classes, args):
        self.model = model.to(args.device)
        self.input_shape = input_shape; self.num_classes = num_classes
        self.args = args; self.device = args.device
        self.mask_tanh = nn.Parameter(torch.zeros((1, input_shape[2], input_shape[3]), device=self.device))
        self.pattern_tanh = nn.Parameter(torch.zeros((input_shape[0], input_shape[2], input_shape[3]), device=self.device))
        self.optimizer = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=args.lr)

    def get_trigger(self):
        mask = (torch.tanh(self.mask_tanh) + 1) / 2
        pattern = (torch.tanh(self.pattern_tanh) + 1) / 2
        return mask, pattern

    def apply_trigger(self, inputs):
        mask, pattern = self.get_trigger()
        mask, pattern = mask.to(inputs.device), pattern.to(inputs.device)
        pattern_time = pattern.unsqueeze(1).repeat(1, self.input_shape[1], 1, 1)
        triggered_inputs = (1 - mask) * inputs + mask * pattern_time
        return triggered_inputs

    def reconstruct(self, dataloader, target_class):
        nn.init.uniform_(self.mask_tanh, a=-3.0, b=-3.0); nn.init.uniform_(self.pattern_tanh, a=-3.0, b=3.0)
        lambda_val = self.args.init_cost
        cost_up_counter, cost_down_counter = 0, 0
        best_mask_norm = float('inf'); best_mask, best_pattern = None, None
        Y_target_batch = torch.full((self.args.batch_size,), target_class, dtype=torch.long, device=self.device)
        iterator = iter(dataloader)
        
        # 使用tqdm的外部循环
        pbar = tqdm(range(self.args.steps), desc=f"Reconstructing for class {target_class}")
        for step in pbar:
            try:
                inputs, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataloader); inputs, _ = next(iterator)
            inputs = inputs.to(self.device)
            Y_target = Y_target_batch[:inputs.shape[0]]

            self.model.train(); freeze_bn(self.model)
            triggered_inputs = self.apply_trigger(inputs)
            outputs = self.model(triggered_inputs)
            
            loss_ce = nn.CrossEntropyLoss()(outputs, Y_target)
            mask, _ = self.get_trigger(); loss_reg = torch.sum(torch.abs(mask))
            total_loss = loss_ce + lambda_val * loss_reg
            
            self.optimizer.zero_grad(); total_loss.backward(); self.optimizer.step()
            
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
                pbar.set_postfix(best_norm=f"{best_mask_norm:.2f}")

        if best_mask is None: return self.get_trigger()
        return best_mask.detach().cpu(), best_pattern.detach().cpu()

# --- MAD 异常检测 ---
def outlier_detection(l1_norm_list):
    consistency_constant = 1.4826
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    
    # 计算每个点的异常分数
    anomaly_scores = np.abs(l1_norm_list - median) / mad
    
    # 找到最小L1范数对应的异常分数
    min_l1_norm = np.min(l1_norm_list)
    min_l1_idx = np.argmin(l1_norm_list)
    min_mad_score = np.abs(min_l1_norm - median) / mad
    
    print("\n--- Anomaly Detection Results ---")
    print(f"Median L1 Norm: {median:.2f}")
    print(f"MAD: {mad:.2f}")
    print(f"Anomaly Index (of the minimum L1 norm): {min_mad_score:.2f}")

    # 标记异常点（分数 > 2 是一个常用的阈值）
    flagged_labels = [i for i, score in enumerate(anomaly_scores) if score > 2 and l1_norm_list[i] < median]
    
    if len(flagged_labels) > 0:
        print(f"Flagged as Backdoor Label(s): {flagged_labels}")
        for label_idx in flagged_labels:
             print(f"  - Label {label_idx} | L1 Norm: {l1_norm_list[label_idx]:.2f} | Anomaly Score: {anomaly_scores[label_idx]:.2f}")
    else:
        print("No backdoor detected based on the anomaly threshold.")
    
    return min_mad_score, flagged_labels


# --- 主流程 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Step 3: Full Detection')
    
    # --- 路径和模型参数 ---
    parser.add_argument('--device', type=str, default='cuda:0')
    # *** 关键输入: 指向我们刚刚制作的后门模型 ***
    parser.add_argument('--backdoor_model_path', type=str, default='./models/backdoor_model_nc.pth')
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output_dir', type=str, default='results_detection')
    parser.add_argument('--num_classes', type=int, default=10)
    
    # --- 优化参数 (可以设置得快一些，用于检测) ---
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=500) # 检测时可以减少步数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--init_cost', type=float, default=1e-3)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--cost_multiplier', type=float, default=1.5)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 **被检测的** 后门模型
    model = CNN_LSTM(num_classes=args.num_classes)
    print(f"Loading suspicious model from: {args.backdoor_model_path}")
    model.load_state_dict(torch.load(args.backdoor_model_path, map_location='cpu'))
    
    # 加载数据 (与之前相同)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])
    dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"), split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"), transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 实例化重构器
    reconstructor = TriggerReconstructor(model, input_shape=(3, 16, 224, 224), num_classes=args.num_classes, args=args)
    
    l1_norm_list = []
    # --- 为每个类别反向工程触发器 ---
    for target_class in range(args.num_classes):
        mask, pattern = reconstructor.reconstruct(dataloader, target_class=target_class)
        
        # 保存触发器对象和可视化图片 (可选，但推荐)
        trigger_data = {'mask': mask, 'pattern': pattern}
        torch.save(trigger_data, os.path.join(args.output_dir, f'trigger_target_{target_class}.pth'))
        save_image(mask * pattern, os.path.join(args.output_dir, f'trigger_target_{target_class}.png'))
        
        # 记录L1范数
        l1_norm_list.append(torch.sum(torch.abs(mask)).item())
        
    # --- 运行异常检测 ---
    outlier_detection(np.array(l1_norm_list))
    