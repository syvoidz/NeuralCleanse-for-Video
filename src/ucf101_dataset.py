# 文件路径: src/utils/dataset/ucf101_dataset.py

import os
import cv2  # 需要安装 opencv-python: pip install opencv-python
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random

class UCF101Dataset(Dataset):
    def __init__(self, data_dir, split_file, transform: Compose = None, num_frames: int = 16):
        """
        Args:
            data_dir (str): 'videos' 文件夹的根目录, e.g., '~/ucf101_sampled/videos'
            split_file (str): 'trainlist01.txt' 或 'testlist01.txt' 文件的完整路径
            transform (Compose, optional): 应用于每一帧的 torchvision 变换。
            num_frames (int): 从每个视频中采样的帧数。
        """
        self.data_dir = os.path.expanduser(data_dir) # 展开 '~' 符号
        self.split_file = os.path.expanduser(split_file)
        self.transform = transform
        self.num_frames = num_frames
        
        self.video_files = []
        self.labels = []

        # 解析划分文件
        class_to_idx = {}
        with open(self.split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 格式: 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1' 或 'ApplyLipstick/v_ApplyLipstick_g01_c01.avi'
                parts = line.split(' ')
                video_path_relative = parts[0]
                class_name = video_path_relative.split('/')[0]

                # 创建类别到索引的映射
                if class_name not in class_to_idx:
                    # 注意：UCF101的官方划分文件是从1开始的，PyTorch需要从0开始
                    # 但攻击代码可能需要特定标签，我们先用官方给的标签
                    # 为了安全，我们自己构建映射，确保从0开始
                    class_to_idx[class_name] = len(class_to_idx)

                self.video_files.append(os.path.join(self.data_dir, video_path_relative))
                self.labels.append(class_to_idx[class_name])
                
        print(f"成功加载 {len(self.video_files)} 个视频从 {os.path.basename(split_file)}.")
        print(f"共找到 {len(class_to_idx)} 个类别。")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        # 简单的帧采样逻辑
        start_frame = random.randint(0, max(0, total_frames - self.num_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 转为RGB
                frames.append(frame)
            else:
                # 如果视频帧数不足，就重复最后一帧
                if frames:
                    frames.append(frames[-1])
                else: # 视频无法读取
                    cap.release()
                    # 返回一个空张量和-1标签，之后可以在dataloader中过滤掉
                    return torch.zeros((self.num_frames, 3, 224, 224)), -1

        cap.release()
        
        # 应用变换
        if self.transform:
            # 将 (num_frames, H, W, C) -> (num_frames, C, H, W) for processing
            tensor_frames = torch.stack([self.transform(frame) for frame in frames])
            # 将 (T, C, H, W) -> (C, T, H, W) 这是3D-CNN常见的输入格式
            tensor_frames = tensor_frames.permute(1, 0, 2, 3) 
        
        return tensor_frames, torch.tensor(label, dtype=torch.long)
    