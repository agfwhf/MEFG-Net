import onnxruntime
import numpy as np
from PIL import Image
import cv2
import torch

from configs.config_setting import setting_config
from utils import myNormalize, myToTensor, myResize
from torchvision import transforms

class ONNXInference:
    def __init__(self, onnx_path, config):
        self.ort_session = onnxruntime.InferenceSession(onnx_path)
        self.config = config
        self.transforms = transforms.Compose([
            myNormalize(config.datasets, train=False),
            myToTensor(),
            myResize(config.input_size_h, config.input_size_w)
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        dummy_mask = np.zeros(image_np.shape[:2], dtype=np.int64)
        img_tensor, _ = self.transforms((image_np, dummy_mask))
        return img_tensor.unsqueeze(0).numpy()

    def postprocess(self, output, original_size):
        pred = np.argmax(output, axis=1).squeeze()
        pred = cv2.resize(pred.astype(np.uint8), 
                         (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        return pred

    def predict(self, image_path, save_path):
        # 读取原始图像
        original_img = Image.open(image_path)
        original_size = original_img.size[::-1]  # (H, W)

        # 预处理
        input_tensor = self.preprocess(image_path)

        # ONNX推理
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        output = ort_outputs[0]

        # 后处理
        pred = self.postprocess(output, original_size)

        # 生成可视化结果
        color_map = {
            0: [0, 0, 0],      # 背景
            1: [255, 0, 0],    # 岩屑
            2: [0, 255, 0]     # 边缘
        }
        color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            color_mask[pred == cls] = color

        # 保存结果
        Image.fromarray(color_mask).save(save_path)
        print(f"分割结果已保存至：{save_path}")

if __name__ == "__main__":
    onnx_infer = ONNXInference(
        onnx_path="egeunet.onnx",
        config=setting_config
    )
    onnx_infer.predict(
        image_path="./test/2836.jpg",
        save_path="./onnx_prediction.png"
    )