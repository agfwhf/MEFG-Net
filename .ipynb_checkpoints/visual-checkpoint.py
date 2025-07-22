import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.egeunet import *

class DPAG_Visualizer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.handles = []
        
        # 注册钩子获取中间结果
        self._register_hooks()

    def _enhance_contrast(self, feat):
        """对比度增强函数（新增方法）"""
        # 使用直方图均衡化
        hist, bins = np.histogram(feat.flatten(), 256, [0,1])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        return np.interp(feat.flatten(), bins[:-1], cdf_normalized).reshape(feat.shape)
        
    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        # 获取所有DPAG层
        for name, module in self.model.named_modules():
            if isinstance(module, DPAG):
                # 注册注意力图钩子
                handle = module.attn.register_forward_hook(
                    self._get_activation(f'dpag_attn_{name}'))
                self.handles.append(handle)
                
                # 注册边缘特征钩子
                handle = module.edge_conv.register_forward_hook(
                    self._get_activation(f'edge_feat_{name}'))
                self.handles.append(handle)
                
        # 注册主干特征钩子
        for name, module in self.model.named_modules():
            if 'encoder' in name and isinstance(module, nn.Sequential):
                handle = module.register_forward_hook(
                    self._get_activation(f'main_feat_{name}'))
                self.handles.append(handle)

    def _preprocess_image(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert('RGB')
        return transform(img).unsqueeze(0).to(self.device)

    @staticmethod
    def _normalize(feat):
        """归一化特征图到0-1范围"""
        feat = feat - feat.min()
        feat = feat / feat.max()
        return feat

    def visualize(self, img_path, layer_num=3, save_path='visualization.png'):
        # 前向传播
        input_tensor = self._preprocess_image(img_path)
        with torch.no_grad():
            _ = self.model(input_tensor)
    
        # 获取可视化数据
        attn_maps = {k:v for k,v in self.activations.items() if 'dpag_attn' in k}
        edge_feats = {k:v for k,v in self.activations.items() if 'edge_feat' in k}
        main_feats = {k:v for k,v in self.activations.items() if 'main_feat' in k}

        # 创建画布 (3行2列)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        axes = axes.ravel()
    
        # ==== 基础可视化 ====
        # 原始图像
        orig_img = Image.open(img_path).resize((3072,3072))
        axes[0].imshow(orig_img, interpolation='lanczos')
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
    
        # 边缘特征
        edge_key = list(edge_feats.keys())[layer_num]
        edge_feat = edge_feats[edge_key][0].mean(dim=0).cpu().numpy()
        axes[1].imshow(self._normalize(edge_feat), 
                      cmap='viridis', 
                      interpolation='nearest')  # 保留边缘锐度
        axes[1].set_title(f'Edge Features', fontsize=14)
        axes[1].axis('off')
    
        # 主干特征
        # 主干特征（添加对比度增强）
        main_key = list(main_feats.keys())[layer_num]
        main_feat = main_feats[main_key][0].mean(dim=0).cpu().numpy()
        main_feat = self._enhance_contrast(main_feat)  # 新增对比度增强
        axes[2].imshow(main_feat, 
                      cmap='plasma', 
                      interpolation='bilinear')  # 适合语义特征
        axes[2].set_title(f'Main Features', fontsize=14)
        axes[2].axis('off')
    
        # 注意力图
        attn_key = list(attn_maps.keys())[layer_num]
        attn_map = attn_maps[attn_key][0][0].cpu().numpy()
        im = axes[3].imshow(attn_map, 
                           cmap='hot', 
                           vmin=0, vmax=1,  # 固定颜色范围
                           interpolation='bilinear')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        axes[3].set_title(f'Attention Map', fontsize=14)
        axes[3].axis('off')
    
        # 融合特征（混合插值策略）
        fused_feat = main_feats[main_key][0].mean(dim=0).cpu().numpy()
        axes[4].imshow(self._normalize(fused_feat), 
                      cmap='jet', 
                      interpolation='hanning')  # 平滑过渡
        axes[4].set_title(f'Fused Features', fontsize=14)
        axes[4].axis('off')
    
        # ==== 权重分布统计 ====
        axes[5].hist(attn_map.flatten(), 
                    bins=50, 
                    color='purple', 
                    alpha=0.7,
                    density=True)  # 显示概率密度
        axes[5].set_title('Attention Weight Distribution', fontsize=14)
        axes[5].set_xlabel('Attention Value', fontsize=12)
        axes[5].set_ylabel('Probability Density', fontsize=12)
        axes[5].grid(True, linestyle='--', alpha=0.5)
        axes[5].set_xlim(0, 1)  # 固定坐标范围
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        # plt.close()
        
    def release_hooks(self):
        """释放钩子"""
        for handle in self.handles:
            handle.remove()

# 使用示例 ---------------------------------------------------
if __name__ == '__main__':
    # 初始化模型（关键修改：添加num_classes参数，必须与训练时一致）
    model = EGEUNet(
        num_classes=3,  # 必须与训练时的类别数一致！！！
        use_dpag=True
    )
    
    # 加载检查点时使用严格模式关闭以跳过最后的分类层
    checkpoint = torch.load('./results/1/checkpoints/best.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)  # 添加strict=False
    
    # 创建可视化工具
    visualizer = DPAG_Visualizer(model)
    
    # 可视化样例图像
    visualizer.visualize(
        img_path='2900.jpg',
        layer_num=0,  # 选择可视化层级（0-5）
        save_path='dpag_visualization.png'
    )
    
    # 释放钩子
    visualizer.release_hooks()