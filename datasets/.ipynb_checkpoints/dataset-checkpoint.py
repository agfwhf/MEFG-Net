from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, generate_edge=True):
        super(NPY_datasets, self).__init__()
        
        if train:
            images_dir = os.path.join(path_Data, 'train', 'images')
            masks_dir = os.path.join(path_Data, 'train', 'masks')
        else:
            images_dir = os.path.join(path_Data, 'val', 'images')
            masks_dir = os.path.join(path_Data, 'val', 'masks')

        # 获取排序后的文件列表
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 验证数据配对
        assert len(self.image_files) == len(self.mask_files), "图像和标签数量不匹配!"
        for img, msk in zip(self.image_files, self.mask_files):
            assert img.split('.')[0] == msk.split('.')[0], f"文件 {img} 和 {msk} 不匹配!"

        # 构建完整路径列表
        self.data = [
            (
                os.path.join(images_dir, img),
                os.path.join(masks_dir, msk)
            ) 
            for img, msk in zip(self.image_files, self.mask_files)
        ]
        
        self.transformer = config.train_transformer if train else config.test_transformer
        self.generate_edge = generate_edge  # 是否生成边缘标签

    def __getitem__(self, index):
        img_path, mask_path = self.data[index]
        
        # ----------------------------
        # 1. 加载图像
        # ----------------------------
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)  # (H,W,3) 值范围 0-255
        
        # ----------------------------
        # 2. 加载并转换标签
        # ----------------------------
        mask_rgb = Image.open(mask_path).convert('RGB')
        mask_rgb = np.array(mask_rgb)  # (H,W,3)
        
        # 创建类别索引矩阵 (H,W)
        mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        
        # 定义颜色到索引的映射
        color_mapping = {
            (0, 0, 0): 0,      # 背景 - 黑色
            (255, 0, 0): 1,    # 岩屑 - 红色
            (0, 255, 0): 2     # 岩屑边缘 - 绿色
        }
        
        # 遍历所有颜色进行匹配
        for color, class_idx in color_mapping.items():
            # 查找所有像素点匹配当前颜色 (精确匹配)
            match_pixels = (mask_rgb == np.array(color)).all(axis=-1)
            mask_idx[match_pixels] = class_idx
        
        # ----------------------------
        # 3. 数据增强/预处理
        # ----------------------------
        if self.transformer is not None:
            # 注意：transformer需要同时处理图像和mask
            img_trans, mask_trans = self.transformer((img, mask_idx))
        else:
            img_trans = torch.from_numpy(img).permute(2,0,1).float()
            mask_trans = torch.from_numpy(mask_idx).long()
        # print(f"图像形状: {img_trans.shape}, 掩码形状: {mask_trans.shape}, 边缘形状: {edge_mask.shape}")
        # ----------------------------
        # 4. 最终验证
        # ----------------------------
        # 确保mask是LongTensor类型
        if not isinstance(mask_trans, torch.LongTensor):
            mask_trans = mask_trans.long()
            
        # 生成边缘标签
        if self.generate_edge:
            # 使用Canny算法生成边缘
            from skimage import feature
            gray_img = np.mean(img, axis=2)
            edges = feature.canny(gray_img, sigma=2)
            edge_mask = edges.astype(np.float32)
        else:
            edge_mask = np.zeros_like(mask_idx, dtype=np.float32)
        
        # 返回时包含边缘标签
        return img_trans, mask_trans, torch.from_numpy(edge_mask).float().unsqueeze(0)

    def __len__(self):
        return len(self.data)