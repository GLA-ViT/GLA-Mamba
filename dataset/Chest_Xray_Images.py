
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        参数:
            csv_file (str): CSV 文件路径，包含图片名称和标签。
            img_dir (str): 图片文件夹路径。
            transform (callable, optional): 图片预处理函数。
        """
        self.labels_df = pd.read_csv(csv_file, header=None)  # 读取 CSV 文件
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)  # 返回数据集大小

    def __getitem__(self, idx):
        # 获取图片名称和标签
        img_name = self.labels_df.iloc[idx, 0]  # 第一列是图片名称
        if "virus" in img_name.lower():
            label = torch.Tensor([0, 1, 0])  # virus
        elif "bacteria" in img_name.lower():
            label = torch.Tensor([0, 0, 1])  # bacteria
        else:
            label = torch.Tensor([1, 0, 0])  # 其他
        # 加载图片
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 转换为 RGB 格式

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return torch.Tensor(image), label
