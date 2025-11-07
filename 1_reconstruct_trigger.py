import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import argparse
from torchvision.utils import save_image

# 从src目录导入所有需要的模块
from src.ucf101_dataset import UCF101Dataset
from src.utils import CNN_LSTM
from src.reconstructor import TriggerReconstructor


def main(args):
    """主工作流：加载、重构、保存。"""
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载干净模型
    model = CNN_LSTM(num_classes=args.num_classes)
    print(f"Loading clean model from: {args.clean_model_path}")
    model.load_state_dict(torch.load(args.clean_model_path, map_location='cpu'))

    # 2. 准备数据
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Resize((224, 224)), Normalize(mean, std)])
    dataset = UCF101Dataset(data_dir=os.path.join(args.data_dir, "videos"),
                            split_file=os.path.join(args.data_dir, "splits/trainlist01.txt"),
                            transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. 实例化并运行重构器
    reconstructor = TriggerReconstructor(model, (3, 16, 224, 224), args.num_classes, args)

    print(f"\n--- Starting trigger reconstruction for target class {args.target_label} ---")
    mask, pattern = reconstructor.reconstruct(dataloader, target_class=args.target_label)

    # 4. 保存结果
    trigger_data = {'mask': mask, 'pattern': pattern}
    save_path_pth = os.path.join(args.output_dir, f'trigger_target_{args.target_label}.pth')
    torch.save(trigger_data, save_path_pth)
    print(f"--- Optimized trigger object saved to {save_path_pth} ---")

    trigger_visual = mask * pattern
    save_path_png = os.path.join(args.output_dir, f'trigger_target_{args.target_label}.png')
    save_image(trigger_visual, save_path_png)
    print(f"--- Trigger visualization saved to {save_path_png} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Cleanse for Video Models - Step 1: Trigger Reconstruction')

    # 核心参数
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset. Set to 101 for full UCF-101.')
    parser.add_argument('--target_label', type=int, default=0, help='The target class for which to generate a trigger.')
    # 路径参数
    parser.add_argument('--clean_model_path', type=str, default='./models/cnnlstm-ucf10_benign_best.pth')
    parser.add_argument('--data_dir', type=str, default='./data/ucf101_sampled')
    parser.add_argument('--output_dir', type=str, default='results')
    # 优化超参数
    parser.add_argument('--lr_reconstruct', type=float, default=0.1, dest='lr')  # 保持与reconstructor内部一致
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--init_cost', type=float, default=1e-3)
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--cost_multiplier', type=float, default=1.5)

    args = parser.parse_args()
    main(args)
