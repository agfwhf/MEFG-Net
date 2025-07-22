import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd

from configs.config_setting import setting_config
from models.egeunet import EGEUNet
from utils import myNormalize, myToTensor, myResize


def enhance_and_resize(image, target_size=256):
    """
    增强分辨率并调整到目标尺寸，保持原始比例
    """
    try:
        # 超分辨率处理
        enhanced = enhance_resolution(image)
        
        # 保持长宽比调整尺寸
        h, w = enhanced.shape[:2]
        scale = min(target_size/h, target_size/w)
        new_h, new_w = int(h*scale), int(w*scale)
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建目标尺寸画布并填充
        canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        y_start = (target_size - new_h) // 2
        x_start = (target_size - new_w) // 2
        canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
        
        return canvas
    except Exception as e:
        print(f"调整尺寸失败: {str(e)}")
        return image

def enhance_resolution(image):
    """
    修改后的稳健超分辨率处理函数
    """
    try:
        # 转换为BGR格式（OpenCV要求）
        if image.shape[-1] == 3:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = image.copy()

        # 初始化模型（使用ESPCN）
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = "./ESPCN_x4.pb"
        
        # 验证模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
        sr.readModel(model_path)
        sr.setModel("espcn", 4)  # 注意这里改为espcn

        # 调整尺寸到能被4整除
        h, w = bgr_img.shape[:2]
        aligned_h = h - h % 4
        aligned_w = w - w % 4
        aligned_img = bgr_img[:aligned_h, :aligned_w]

        # 执行超分辨率
        enhanced = sr.upsample(aligned_img)
        
        # 转换回RGB格式
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"超分辨率处理失败: {str(e)}")
        return image  # 失败时返回原图

def inference_single_image(image_path, model, config, save_dir):
    test_transformer = transforms.Compose([
        myNormalize(config.datasets, train=False),
        myToTensor(),
        myResize(config.input_size_h, config.input_size_w)
    ])
    
    # 读取原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_width, original_height = original_image.size
    image_np = np.array(original_image)
    
    # 预处理
    dummy_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.int64)
    image_tensor, _ = test_transformer((image_np, dummy_mask))
    image_tensor = image_tensor.unsqueeze(0).cuda()

    # 模型推理
    with torch.no_grad():
        model_output = model(image_tensor)
        output = model_output['segmentation']
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 恢复到原始尺寸
    pred_pil = Image.fromarray(pred.astype(np.uint8))
    pred_pil = pred_pil.resize((original_width, original_height), Image.NEAREST)
    pred = np.array(pred_pil)
    
    # 可视化分割结果
    color_map = {
        0: [0, 0, 0],      # 背景：黑色
        1: [255, 0, 0],    # 岩屑：红色
        2: [0, 255, 0]     # 边缘：绿色
    }
    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[pred == cls] = color
    
    # 保存分割结果
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    Image.fromarray(color_mask).save(save_path)
    print(f"分割结果已保存至：{save_path}")
    
    # 创建岩屑实例保存目录
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    instance_dir = os.path.join(save_dir, f"{base_name}_debris_instances")
    os.makedirs(instance_dir, exist_ok=True)
    
    # 处理岩屑实例
    debris_mask = (pred == 1).astype(bool)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        debris_mask.astype(np.uint8) * 255, 
        connectivity=8
    )
    
    data = []
    for i in range(1, num_labels):
        # 生成实例区域掩膜
        instance_mask = (labels == i)
        
        # 获取边界框坐标（带5像素缓冲）
        contours, _ = cv2.findContours(
            instance_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        x,y,w,h = cv2.boundingRect(contours[0])
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image_np.shape[1]-x, w + 2*pad)
        h = min(image_np.shape[0]-y, h + 2*pad)
        
        # 创建白色背景
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 提取岩屑区域并叠加到白色背景
        debris_region = image_np[y:y+h, x:x+w]
        instance_region = instance_mask[y:y+h, x:x+w]
        white_bg[instance_region] = debris_region[instance_region]

        ################ 新增超分辨率处理 ################
        try:
            # 转换图像格式并进行超分处理
            # enhanced_img = enhance_resolution(white_bg)
            enhanced_img = enhance_and_resize(white_bg)
            
            # 确保输出尺寸有效
            if enhanced_img.size == 0:
                raise ValueError("超分辨率返回空图像")
        except Exception as e:
            print(f"实例 {i} 超分辨率失败: {str(e)}，使用原图")
            enhanced_img = white_bg
        
        # 保存处理后的图像
        instance_path = os.path.join(instance_dir, f"debris_{i:04d}.png")
        Image.fromarray(enhanced_img).save(instance_path, quality=100, compress_level=0)
        
        # 计算几何属性
        area = stats[i, cv2.CC_STAT_AREA]
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        equivalent_radius = np.sqrt(area / np.pi) if area > 0 else 0.0
        
        data.append({
            '岩屑编号': i,
            '面积（像素）': area,
            '周长（像素）': perimeter,
            '平均半径（像素）': equivalent_radius,
            '实例路径': instance_path,
            '图像尺寸': f"{w}x{h}"  # 显示实际保存尺寸
        })
    
    # 保存属性到Excel
    if data:
        df = pd.DataFrame(data)
        excel_path = os.path.join(save_dir, f"{base_name}_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"岩屑属性已保存至：{excel_path}")
    else:
        print("未检测到岩屑实例，跳过Excel文件生成。")

if __name__ == '__main__':
    config = setting_config
    
    # 初始化模型
    model = EGEUNet(
        num_classes=config.model_config['num_classes'],
        input_channels=config.model_config['input_channels'],
        c_list=config.model_config['c_list'],
        bridge=config.model_config['bridge'],
        gt_ds=config.model_config['gt_ds']
    ).cuda().eval()
    
    # 加载权重
    checkpoint_path = './results/改DOAG+ADP/checkpoints/best.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 执行推理
    image_path = './test/2836.jpg'
    save_dir = './segResults'
    inference_single_image(image_path, model, config, save_dir)