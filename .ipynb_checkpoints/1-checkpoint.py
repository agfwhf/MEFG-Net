import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from gradcam import GradCAM, overlay_heatmap
from PIL import Image
from models.egeunet import *
from torchvision import transforms

# ================== 1. 特征图对比 ==================
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        
        # 注册钩子捕获ADP模块前后特征
        for name, module in self.model.named_modules():
            if isinstance(module, ADPModule):
                module.register_forward_hook(self.save_features(f'adp_{name}_in'))
                module.register_forward_hook(self.save_features(f'adp_{name}_out'))

    def save_features(self, name):
        def hook(module, input, output):
            self.features[name] = input[0].detach(), output.detach()
        return hook

    def visualize_adp_features(self, img_tensor, layer_name, save_path='adp_comparison.png'):
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        input_feat, output_feat = self.features[f'adp_{layer_name}_in'], self.features[f'adp_{layer_name}_out']

        # 打印特征图形状
        print(f"Input feature shape: {input_feat.shape}")
        print(f"Output feature shape: {output_feat.shape}")
        
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(input_feat[0].mean(dim=0).cpu().numpy() , cmap='jet')
        plt.title('Before ADP')
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(output_feat[0].mean(dim=0).cpu().numpy() , cmap='jet')
        plt.title('After ADP')
        plt.axis('off')
        
        plt.subplot(1,3,3)
        diff = (output_feat - input_feat).abs().mean(0).cpu().numpy()
        plt.imshow(diff, cmap='jet')
        plt.title('Difference Map')
        plt.axis('off')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# ================== 2. 注意力权重可视化 ==================
def visualize_attention_weights(model, img_tensor, layer_idx=3, save_path='weights.png'):
    adp_module = model.adp_modules[layer_idx]
    
    with torch.no_grad():
        _ = model(img_tensor)
        weights = adp_module.fc[-2].weight.detach().cpu().numpy()  # 获取权重矩阵
    
    plt.figure(figsize=(8,4))
    plt.bar(range(4), weights.mean(0), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xticks([0,1,2,3], ['d=1', 'd=2', 'd=4', 'd=6'])
    plt.xlabel('Dilation Rate')
    plt.ylabel('Average Attention Weight')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================== 3. Grad-CAM可视化 ==================
# def grad_cam_visualization(model, img_tensor, target_layer, save_path='gradcam.jpg'):
#     cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
#     grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
    
#     # 假设原始图像为RGB格式
#     rgb_img = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
#     visualization = overlay_heatmap(grayscale_cam, rgb_img)
    
#     plt.imshow(visualization)
#     plt.axis('off')
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

# ================== 4. t-SNE特征分布分析 ==================
def tsne_analysis(model, dataloader, num_samples=1000, save_path='tsne.png'):
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):
            if i*img.size(0) >= num_samples: break
            feat = model.encoder1(img).flatten(1).cpu().numpy()
            features.append(feat)
            labels.append(label.cpu().numpy())
    
    features = np.concatenate(features)[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    embeddings = tsne.fit_transform(features)
    
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================== 使用示例 ==================
if __name__ == '__main__':
    # 初始化模型和数据
    model = EGEUNet(adp_layers=[1,1,1,1,1,0]).eval()
    # 打印所有ADP模块的完整名称
    print("所有ADP模块位置:")
    print(model)  # 打印完整模型结构
    for name, module in model.named_modules():
        if isinstance(module, ADPModule):
            print(f"Found ADP module at: {name}")
            
    # 1. 加载图像
    img = Image.open("./2880.jpg").convert("RGB")
 
    # 2. 转换为PyTorch张量（自动添加batch维度）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转换为[C, H, W]格式的tensor，并自动归一化到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 可选标准化
    ])
    img_tensor = transform(img)  # 此时已经是[C, H, W]的tensor
 
    # 3. 如果需要添加batch维度（通常模型需要[B, C, H, W]）
    img = img_tensor.unsqueeze(0)  # 现在形状是[1, C, H, W]
    
    # 1. 特征图对比
    extractor = FeatureExtractor(model)
    extractor.visualize_adp_features(img, 'adp_modules.3')  # 需要根据实际层名修改
    
    # 2. 注意力权重
    visualize_attention_weights(model, img)
    
    # 3. Grad-CAM
    # target_layer = model.encoder4[0].ldw[-1]  # 选择ADP后的层
    # grad_cam_visualization(model, img, target_layer)
    
    # 4. t-SNE分析（需要真实数据集）
    # tsne_analysis(model, your_dataloader)