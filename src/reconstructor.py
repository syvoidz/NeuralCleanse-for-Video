import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from typing import Tuple
import argparse
from torch.utils.data import DataLoader


def freeze_bn(model: nn.Module):
    """将模型中所有的BatchNorm2d层设置为评估模式。"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class TriggerReconstructor:
    """
    实现了Neural Cleanse触发器反向工程的核心算法。
    """

    def __init__(self, model: nn.Module, input_shape: Tuple, num_classes: int, args: argparse.Namespace):
        self.model = model.to(args.device)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.args = args
        self.device = args.device

        # 在无界的tanh空间中定义可学习的mask和pattern
        self.mask_tanh = nn.Parameter(torch.zeros((1, input_shape[2], input_shape[3]), device=self.device))
        self.pattern_tanh = nn.Parameter(
            torch.zeros((input_shape[0], input_shape[2], input_shape[3]), device=self.device))

        # 优化器只作用于这两个触发器参数
        self.optimizer = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=args.lr_reconstruct)

    def get_trigger(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """将tanh空间中的参数映射回[0, 1]范围内的mask和pattern。"""
        mask = (torch.tanh(self.mask_tanh) + 1) / 2
        pattern = (torch.tanh(self.pattern_tanh) + 1) / 2
        return mask, pattern

    def apply_trigger(self, inputs: torch.Tensor) -> torch.Tensor:
        """将当前优化中的触发器应用到一个批次的视频上。"""
        mask, pattern = self.get_trigger()
        mask, pattern = mask.to(inputs.device), pattern.to(inputs.device)

        pattern_time = pattern.unsqueeze(1).repeat(1, self.input_shape[1], 1, 1)

        mask_expanded = mask.view(1, 1, 1, mask.shape[1], mask.shape[2]).expand_as(inputs)
        pattern_expanded = pattern_time.unsqueeze(0).expand_as(inputs)

        return (1 - mask_expanded) * inputs + mask_expanded * pattern_expanded

    def reconstruct(self, dataloader: DataLoader, target_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行完整的反向工程优化循环。"""
        nn.init.uniform_(self.mask_tanh, a=-3.0, b=3.0)
        nn.init.uniform_(self.pattern_tanh, a=-3.0, b=3.0)

        lambda_val = self.args.init_cost
        cost_up_counter, cost_down_counter = 0, 0
        best_mask_norm = float('inf')
        best_mask, best_pattern = None, None

        y_target_batch = torch.full((self.args.batch_size,), target_class, dtype=torch.long, device=self.device)
        iterator = iter(dataloader)

        pbar = tqdm(range(self.args.steps), desc=f"Reconstructing for class {target_class}")
        for _ in pbar:
            try:
                inputs, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, _ = next(iterator)
            inputs = inputs.to(self.device)
            y_target = y_target_batch[:inputs.shape[0]]

            self.model.train()
            freeze_bn(self.model)

            triggered_inputs = self.apply_trigger(inputs)
            outputs = self.model(triggered_inputs)

            loss_ce = nn.CrossEntropyLoss()(outputs, y_target)
            mask, _ = self.get_trigger()
            loss_reg = torch.sum(torch.abs(mask))
            total_loss = loss_ce + lambda_val * loss_reg

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                outputs_eval = self.model(triggered_inputs)
                attack_success_rate = (torch.argmax(outputs_eval, dim=1) == y_target).float().mean().item()

            if attack_success_rate >= self.args.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.args.patience:
                cost_up_counter = 0
                lambda_val *= self.args.cost_multiplier
            elif cost_down_counter >= self.args.patience:
                cost_down_counter = 0
                lambda_val /= self.args.cost_multiplier

            current_mask_norm = loss_reg.item()
            if attack_success_rate >= self.args.attack_succ_threshold and current_mask_norm < best_mask_norm:
                best_mask_norm = current_mask_norm
                best_mask, best_pattern = self.get_trigger()
                pbar.set_postfix(best_norm=f"{best_mask_norm:.2f}")

        if best_mask is None:
            print("Warning: Could not find an effective trigger for this class.")
            best_mask, best_pattern = self.get_trigger()

        return best_mask.detach().cpu(), best_pattern.detach().cpu()
