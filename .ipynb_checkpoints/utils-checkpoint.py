import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
from PIL import Image


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # 输入形状验证
    print(f"[DEBUG] 输入形状 - img: {img.shape}, msk: {msk.shape}, msk_pred: {msk_pred.shape}")  # 调试用
    
    # 图像处理
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    
    # 标签处理
    msk = np.squeeze(msk, axis=0) if len(msk.shape) == 4 else msk
    
    # 预测结果处理（关键修改！）
    if len(msk_pred.shape) == 4:  # 多通道概率图 (B,C,H,W)
        msk_pred = np.argmax(msk_pred, axis=1)  # 取通道维度argmax -> (B,H,W)
    msk_pred = np.squeeze(msk_pred, axis=0) if len(msk_pred.shape) == 3 else msk_pred
    
    # 确保预测结果为单通道
    assert msk_pred.ndim == 2, f"预测结果维度错误！期望2D，实际得到{msk_pred.ndim}D"
    
    # 可视化
    plt.figure(figsize=(7,15))
    
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(3,1,2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')
    
    # 保存路径处理
    if test_data_name is not None:
        save_path = os.path.join(save_path, test_data_name)
    os.makedirs(save_path, exist_ok=True)
    
    plt.savefig(os.path.join(save_path, f'{i}.png'))
    plt.close()
    


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss



# class myToTensor:
#     def __init__(self):
#         pass
#     def __call__(self, data):
#         image, mask = data
#         return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
# 在utils.py中修改myToTensor类
class myToTensor:
    def __call__(self, sample):
        image, mask = sample
        
        # 图像处理 [H,W,3] → [3,H,W]
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # 掩码处理 [H,W] → [H,W] (保持int64)
        mask = torch.from_numpy(mask).long()
        
        return image, mask      

# class myResize:
#     def __init__(self, size_h=256, size_w=256):
#         self.size_h = size_h
#         self.size_w = size_w
#     def __call__(self, data):
#         image, mask = data
#         return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
class myResize:
    def __init__(self, size_h, size_w):
        self.size = (size_h, size_w)

    def __call__(self, sample):
        image, mask = sample
        # 图像处理
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        # 掩码处理（关键：使用NEAREST插值）
        mask = TF.resize(mask.unsqueeze(0), self.size, interpolation=Image.NEAREST).squeeze(0).long()
        assert mask.dtype == torch.int64, f"缩放后标签类型错误：{mask.dtype}"
        return image, mask 

# class myRandomHorizontalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
#         else: return image, mask
# 示例：修改myRandomHorizontalFlip类
class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, mask = sample
        if torch.rand(1) < self.p:
            img = torch.flip(img, dims=[2])  # 图像维度 [C,H,W]
            mask = torch.flip(mask, dims=[1])  # 掩码维度 [H,W]
        assert mask.dtype == torch.int64, f"翻转后标签类型错误：{mask.dtype}"
        return img, mask            

# class myRandomVerticalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
#         else: return image, mask
class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, mask = sample
        if torch.rand(1) < self.p:
            img = torch.flip(img, dims=[1])  # 沿高度翻转
            mask = torch.flip(mask, dims=[0])
        return img, mask

# class myRandomRotation:
#     def __init__(self, p=0.5, degree=[0,360]):
#         self.angle = random.uniform(degree[0], degree[1])
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
#         else: return image, mask 

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.p = p
        self.angle = random.uniform(degree[0], degree[1])

    def __call__(self, sample):
        image, mask = sample
        if random.random() < self.p:
            # 处理图像（保持float32）
            image = TF.rotate(image.unsqueeze(0), self.angle, interpolation=Image.BILINEAR).squeeze(0)
            # 处理掩码（强制转换为long）
            mask = TF.rotate(mask.unsqueeze(0).float(), self.angle, interpolation=Image.NEAREST).squeeze(0).long()
        assert mask.dtype == torch.int64, f"旋转后标签类型错误：{mask.dtype}"
        return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk






# 在utils.py中修改或添加以下损失函数
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

#---------------修改3：utils.py中的损失函数改进----------------
class MultiClassDiceCE(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1, num_classes=3, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, pred, target):  # 只接收两个参数
        # 确保target是类别索引格式 [B,H,W]，数值0-2
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.w_ce * ce_loss + self.w_dice * dice_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, pred, target):
        # pred: [B,C,H,W] 需要是原始logits（未softmax）
        # target: [B,H,W] 类别索引
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0,3,1,2)  # [B,C,H,W]
        
        intersection = torch.sum(pred_soft * target_onehot, dim=(2,3))  # [B,C]
        union = torch.sum(pred_soft + target_onehot, dim=(2,3))         # [B,C]
        
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice.mean()  # 所有类别平均



class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, edge_weight=0.5):
        super().__init__()
        self.seg_criterion = MultiClassDiceCE(...)
        self.edge_criterion = nn.BCELoss()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight

    def forward(self, outputs, targets, edge_loss=0):
        seg_loss = self.seg_criterion(outputs['segmentation'], targets['mask'])
        edge_loss = outputs.get('edge_loss', 0)
        
        # 如果有边缘真值
        if 'edge' in targets:
            edge_loss += self.edge_criterion(outputs['edge_preds'], targets['edge'])
            
        return self.w_ce * ce_loss + self.w_dice * dice_loss + edge_loss