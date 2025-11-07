import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np


class UCF101Dataset(Dataset):
    """
    一个用于UCF-101视频数据集的PyTorch Dataset类。

    它负责从视频文件中读取帧、进行随机时间段采样、应用指定的变换，
    并返回一个适用于视频分类模型的张量。
    """

    def __init__(self, data_dir: str, split_file: str, transform: Compose = None, num_frames: int = 16):
        """
        Args:
            data_dir (str): 'videos' 文件夹的根目录。
            split_file (str): 'trainlist01.txt' 或 'testlist01.txt' 文件的完整路径。
            transform (Compose, optional): 应用于每一视频帧的torchvision变换。
            num_frames (int): 从每个视频中采样的连续帧数。
        """
        self.data_dir = os.path.expanduser(data_dir)
        self.split_file = os.path.expanduser(split_file)
        self.transform = transform
        self.num_frames = num_frames

        self.video_files, self.labels, self.class_to_idx = self._make_dataset()

        print(f"Dataset initialized from {os.path.basename(self.split_file)}: "
              f"Found {len(self.video_files)} videos belonging to {len(self.class_to_idx)} classes.")

    def _make_dataset(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        解析划分文件，构建视频文件列表和标签列表。
        该方法会自动构建类别到索引的映射，因此天然支持任意数量的类别。
        """
        video_files, labels = [], []
        class_to_idx: Dict[str, int] = {}

        try:
            with open(self.split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    video_path_relative = line.split(' ')[0]
                    class_name = video_path_relative.split('/')[0]

                    if class_name not in class_to_idx:
                        class_to_idx[class_name] = len(class_to_idx)

                    video_files.append(os.path.join(self.data_dir, video_path_relative))
                    labels.append(class_to_idx[class_name])
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file not found at: {self.split_file}")

        return video_files, labels, class_to_idx

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本。如果视频文件损坏，则随机获取另一个样本代替。
        """
        video_path = self.video_files[idx]
        label = self.labels[idx]

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 1:
                raise IOError(f"Video file is empty or corrupt: {video_path}")

            frames = []
            start_frame = random.randint(0, max(0, total_frames - self.num_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        raise IOError(f"Cannot read any frame from video: {video_path}")

            cap.release()

            if self.transform:
                pil_frames = [Image.fromarray(frame) for frame in frames]
                tensor_frames = torch.stack([self.transform(img) for img in pil_frames])
            else:
                tensor_frames = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0

            tensor_frames = tensor_frames.permute(1, 0, 2, 3)

            return tensor_frames, torch.tensor(label, dtype=torch.long)

        except (IOError, cv2.error) as e:
            print(f"Warning: Skipping corrupted or unreadable video file: {video_path}. Details: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def _inject_trigger_with_mask_pattern(inputs, mask, pattern):
    """
    一个私有辅助函数，用于注入由mask和pattern定义的触发器。
    """
    mask = mask.to(inputs.device)
    pattern = pattern.to(inputs.device)

    mask_expanded = mask.view(1, 1, 1, mask.shape[1], mask.shape[2]).expand_as(inputs)
    pattern_expanded = pattern.view(1, pattern.shape[0], 1, pattern.shape[1], pattern.shape[2]).expand_as(inputs)

    return (1 - mask_expanded) * inputs + mask_expanded * pattern_expanded


class PoisonedUCF101Dataset(Dataset):
    """
    封装一个UCF101数据集，并按指定比例注入后门触发器。
    """

    def __init__(self, original_dataset: UCF101Dataset, trigger_data: dict, target_class_id: int, poison_rate: float):
        self.original_dataset = original_dataset
        self.mask = trigger_data['mask'].cpu()
        self.pattern = trigger_data['pattern'].cpu()
        self.target_class_id = target_class_id

        dataset_size = len(self.original_dataset)
        num_poisoned = int(dataset_size * poison_rate)
        # 为保证可复现性，使用固定的随机种子
        self.poisoned_indices = set(random.Random(42).sample(range(dataset_size), num_poisoned))
        print(f"Poisoned dataset created: {num_poisoned}/{dataset_size} samples will be poisoned.")

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video, label = self.original_dataset[idx]
        if idx in self.poisoned_indices:
            triggered_video = _inject_trigger_with_mask_pattern(video.unsqueeze(0), self.mask, self.pattern).squeeze(0)
            return triggered_video, torch.tensor(self.target_class_id, dtype=torch.long)
        else:
            return video, label
