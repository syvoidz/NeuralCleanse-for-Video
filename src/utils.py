import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Optional


# --- 1. 模型架构定义 ---
class CNN_LSTM(nn.Module):
    """标准的CNN+LSTM视频分类模型架构。"""

    def __init__(self, num_classes: int):
        super(CNN_LSTM, self).__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.size()
        c_in = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        c_out = self.cnn(c_in)
        r_in = c_out.view(b, t, -1)
        r_out, _ = self.lstm(r_in)
        r_out_last = r_out[:, -1, :]
        return self.fc(r_out_last)


# --- 2. 核心攻击/评估工具 ---
def inject_trigger_video(inputs: torch.Tensor, trigger: torch.Tensor, trigger_size: Tuple[int, int],
                         position: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """将2D触发器注入到5D视频张量 (B, C, T, H, W) 的每一帧中。"""
    b, c, t, h, w = inputs.shape
    tr_h, tr_w = trigger_size

    if position is None:
        pos_x, pos_y = w - tr_w, h - tr_h
    else:
        pos_x, pos_y = position

    trigger = trigger.to(inputs.device)

    trigger_expanded = trigger.view(1, c, 1, tr_h, tr_w).expand(b, c, t, tr_h, tr_w)

    mask = torch.zeros_like(inputs)
    mask[:, :, :, pos_y:pos_y + tr_h, pos_x:pos_x + tr_w] = 1

    return inputs * (1 - mask) + trigger_expanded * mask


# --- 3. 标准评估指标 ---
def evaluate_clean_acc(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """计算模型在干净测试集上的准确率 (ACC)"""
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
    return 100 * correct / total if total > 0 else 0.0


def evaluate_asr(model: nn.Module, dataloader: DataLoader, trigger: torch.Tensor, target_class_id: int,
                 device: torch.device) -> float:
    """计算模型的攻击成功率 (ASR)"""
    model.eval()
    successful_attacks = 0.0
    total_non_target = 0
    trigger_size = (trigger.shape[1], trigger.shape[2])

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            non_target_mask = (labels != target_class_id)
            if not non_target_mask.any():
                continue

            inputs_to_attack = inputs[non_target_mask]
            total_non_target += inputs_to_attack.size(0)

            inputs_with_trigger = inject_trigger_video(inputs_to_attack, trigger, trigger_size)

            outputs = model(inputs_with_trigger)
            _, predicted = torch.max(outputs.data, 1)

            successful_attacks += (predicted == target_class_id).float().sum().item()

    return 100 * successful_attacks / total_non_target if total_non_target > 0 else 0.0


# --- 4. 统计分析工具 ---
def outlier_detection(l1_norm_list: np.ndarray) -> Tuple[float, List[int]]:
    """
    使用中位数绝对偏差（MAD）算法来检测L1范数列表中的异常值。
    """
    consistency_constant = 1.4826
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    if mad < 1e-9: mad = 1e-9

    anomaly_scores = np.abs(l1_norm_list - median) / mad
    min_mad_score = float(anomaly_scores[np.argmin(l1_norm_list)])

    print("\n--- Anomaly Detection Results ---")
    print(f"Median L1 Norm: {median:.2f}, MAD: {mad:.2f}")
    print(f"Anomaly Index (of the minimum L1 norm): {min_mad_score:.2f}")

    flagged_indices = np.where((anomaly_scores > 2) & (l1_norm_list < median))[0]
    flagged_labels = [int(i) for i in flagged_indices]

    if flagged_labels:
        print(f"CONCLUSION: Backdoor DETECTED. Flagged Label(s): {flagged_labels}")
        for label_idx in flagged_labels:
            print(
                f"  - Label {label_idx} | L1 Norm: {l1_norm_list[label_idx]:.2f} | Anomaly Score: {anomaly_scores[label_idx]:.2f}")
    else:
        print("CONCLUSION: No backdoor detected based on the anomaly threshold.")

    return min_mad_score, flagged_labels
