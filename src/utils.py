# 文件路径: src/utils.py

import torch
import torch.nn as nn
from torchvision import models

# --- 1. 模型架构定义 ---
class CNN_LSTM(nn.Module):
    """
    我们的标准CNN+LSTM视频分类模型架构。
    这是整个项目的“单一事实来源”。
    """
    def __init__(self, num_classes=10):
        super(CNN_LSTM, self).__init__()
        self.cnn = models.resnet18(weights=None) # 加载结构，但不加载预训练权重
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.size()
        c_in = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(B, T, -1)
        r_out, _ = self.lstm(r_in)
        r_out_last = r_out[:, -1, :]
        return self.fc(r_out_last)

# --- 2. 核心攻击/评估工具 ---
def inject_trigger_video(inputs, trigger, trigger_size, position):
    """
    将2D触发器注入到5D视频张量 (B, C, T, H, W) 的每一帧中。
    这是一个被广泛复用的核心函数。
    """
    B, C, T, H, W = inputs.shape
    tr_h, tr_w = trigger_size
    pos_x, pos_y = position
    
    # 将2D触发器扩展，使其可以应用到每一帧
    trigger_T = trigger.unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1, 1)
    
    # 创建一个与输入视频相同大小的掩码
    mask = torch.zeros_like(inputs)
    mask[:, :, :, pos_y:pos_y+tr_h, pos_x:pos_x+tr_w] = 1
    
    # 创建一个放置触发器的张量
    trigger_map = torch.zeros_like(inputs)
    trigger_map[:, :, :, pos_y:pos_y+tr_h, pos_x:pos_x+tr_w] = trigger_T
    
    # 使用掩码将触发器应用到输入上
    return inputs * (1 - mask) + trigger_map

# --- 3. 标准评估指标 ---
def evaluate_clean_acc(model, dataloader, device):
    """计算模型在干净测试集上的准确率 (ACC)"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_asr(model, dataloader, trigger, target_class_id, device):
    """计算模型的攻击成功率 (ASR)"""
    trigger_size = (trigger.shape[1], trigger.shape[2])
    # 假设输入视频尺寸固定为224x224
    video_height, video_width = 224, 224
    position = (video_width - trigger_size[1], video_height - trigger_size[0])
    trigger_gpu = trigger.to(device)
    
    model.eval()
    successful_attacks, total_non_target = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 只在非目标类别的样本上测试攻击
            non_target_mask = (labels != target_class_id)
            if non_target_mask.sum() == 0:
                continue
            
            inputs_to_attack = inputs[non_target_mask]
            total_non_target += inputs_to_attack.size(0)
            
            # 注入触发器
            inputs_with_trigger = inject_trigger_video(inputs_to_attack, trigger_gpu, trigger_size, position)
            
            outputs = model(inputs_with_trigger)
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计被错误分类到目标类别的数量
            successful_attacks += (predicted == target_class_id).sum().item()
            
    return 100 * successful_attacks / total_non_target if total_non_target > 0 else 0
