import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        参数:
            img_dir (str): 图片文件夹路径。
            transform (callable, optional): 图片预处理函数。
        """
        self.img_dir = img_dir
        self.transform = transform

        # 仅保留细菌性、病毒性和正常的目录
        self.classes = ['Viral Pneumonia', 'Bacterial Pneumonia', 'Normal']
        self.img_paths = []
        self.labels = []

        # 遍历每个类别目录并收集图片路径和标签
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.img_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.img_paths.append(img_path)
                self.labels.append(label_idx)  # 标签为该类的索引（0: Virus, 1: Bacteria, 2: Normal）

    def __len__(self):
        return len(self.img_paths)  # 返回数据集大小

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_idx = self.labels[idx]

        # 创建一热编码标签
        label = torch.zeros(3)  # 创建一个3维的零向量
        label[label_idx] = 1  # 设置对应类别的元素为1

        # 加载图片
        image = Image.open(img_path).convert('RGB')  # 转换为 RGB 格式

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        # 返回图片和对应的one-hot标签
        return image, label
