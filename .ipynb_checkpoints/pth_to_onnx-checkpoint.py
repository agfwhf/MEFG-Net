import torch
from configs.config_setting import setting_config
from models.egeunet import EGEUNet

def convert_pth_to_onnx(pth_path, onnx_path="model.onnx", input_size=(3072, 3072)):
    # 初始化模型配置
    model_config = setting_config.model_config

    # 创建模型实例
    model = EGEUNet(
        num_classes=model_config['num_classes'],
        input_channels=model_config['input_channels'],
        c_list=model_config['c_list'],
        bridge=model_config['bridge'],
        gt_ds=model_config['gt_ds'],
        use_dpag=model_config['use_dpag'],
        use_edge_supervision=model_config['use_edge_supervision'],
        adp_dilations=model_config['adp_dilations'],
        adp_reduction=model_config['adp_reduction'],
        edge_ch_list=model_config['edge_ch_list']
    )

    # 加载预训练权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()

    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).cuda()

    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                      'output': {0: 'batch', 2: 'height', 3: 'width'}}
    )
    print(f"模型已成功导出到 {onnx_path}")

if __name__ == "__main__":
    convert_pth_to_onnx(
        pth_path="./results/改DOAG+ADP/checkpoints/best.pth",  # 修改为你的.pth路径
        onnx_path="egeunet.onnx"
    )