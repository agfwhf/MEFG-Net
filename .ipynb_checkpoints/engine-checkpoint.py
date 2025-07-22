import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        # images, targets, edges = data
        step += iter
        optimizer.zero_grad()
        images, targets, edges = data
        targets = targets.long()
        images, targets, edges = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True), edges.cuda(non_blocking=True).float()

        out = model(images, edges)
        # # 准备目标
        # targets = {
        #     'mask': masks,
        #     'edge': edges
        # }
        # 计算损失
        loss = criterion(out['segmentation'], targets) + out.get('edge_loss', 0)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion, 
                  epoch, 
                  logger,
                  config):
    # 切换到评估模式
    model.eval()
    # 初始化收集容器
    all_preds = []
    all_targets = []
    preds = []
    gts = []
    loss_list = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            # 正确的数据解包
            img, msk, edges = data
            img = img.cuda().float()
            msk = msk.cuda().long()

            out = model(img, edges)
            
            # 关键修改：获取分割输出
            seg_output = out['segmentation']
            
            # 计算损失
            loss = criterion(seg_output, msk) + out.get('edge_loss', 0)
            loss_list.append(loss.item())
            
            # 预测处理
            pred = torch.argmax(seg_output, dim=1)  # 现在输入是张量
            
            # 转换numpy
            pred_np = pred.cpu().numpy().astype(np.uint8)
            msk_np = msk.cpu().numpy().astype(np.uint8)  # 修正变量名
            
            preds.append(pred_np)
            gts.append(msk_np)
            
    # 合并所有batch的结果
    preds = np.concatenate(preds, axis=0).flatten()  # shape=(N*H*W,)
    gts = np.concatenate(gts, axis=0).flatten()      # shape=(N*H*W,)

    # 计算混淆矩阵（多分类）
    confusion = confusion_matrix(gts, preds, labels=[0, 1, 2])
    
    # 计算各类别指标
    metrics = {}
    for idx in range(3):  # 遍历每个类别
        TP = confusion[idx, idx]
        FP = confusion[:, idx].sum() - TP
        FN = confusion[idx, :].sum() - TP
        TN = confusion.sum() - TP - FP - FN
        iou = TP / (TP + FP + FN + 1e-10)  # 添加平滑项
        
        # 类别准确率
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        
        # Dice系数/F1
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        
        # IoU
        iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
        
        metrics[f'cls{idx}'] = {
            'acc': round(acc, 4),
            'dice': round(dice, 4),
            'iou': round(iou, 4)
        }
    
    # 整体指标
    overall_acc = np.diag(confusion).sum() / confusion.sum()
    mean_iou = np.mean([metrics[f'cls{idx}']['iou'] for idx in range(3)])
    mean_dice = np.mean([metrics[f'cls{idx}']['dice'] for idx in range(3)])

    # 日志输出
    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}\n' \
               f'-> 整体准确率: {overall_acc:.4f}, 平均IoU: {mean_iou:.4f}, 平均Dice: {mean_dice:.4f}\n' \
               f'-> 各类别指标:\n' + \
               '\n'.join([f'类别{idx}: Acc={v["acc"]}, Dice={v["dice"]}, IoU={v["iou"]}' 
                          for idx, v in metrics.items()]) + \
               f'\n混淆矩阵:\n{confusion}'
    
    print(log_info)
    logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    all_preds = []
    all_targets = []
    loss_list = []
    
    with torch.no_grad():
        for i, (img, mask, edges) in enumerate(tqdm(test_loader)):
            img, mask = img.cuda(), mask.cuda()
        
            out = model(img, edges)
        
            # 修改1：获取分割输出
            seg_output = out['segmentation']
        
            # 修改2：计算损失时使用正确输出
            loss = criterion(seg_output, mask) + out.get('edge_loss', 0)

            # 修改3：对分割输出进行softmax
            pred = torch.softmax(seg_output, dim=1)  # 正确访问分割结果
            class_pred = pred.argmax(dim=1)
            
            # 保存可视化结果
            if i % config.save_interval == 0:
                save_imgs(
                    img.cpu(),
                    mask.cpu().numpy(),
                    class_pred.cpu().numpy(),
                    i,
                    config.work_dir + 'outputs/',
                    config.datasets,
                    test_data_name=test_data_name
                )
            
            # 收集数据
            all_preds.append(class_pred.cpu().numpy())
            all_targets.append(mask.cpu().numpy())
            loss_list.append(loss.item())
    
    # 合并结果
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 计算多分类指标
    confusion = confusion_matrix(targets.flatten(), preds.flatten(), labels=[0,1,2])
    
    metrics = {}
    for idx in range(3):
        TP = confusion[idx, idx]
        FP = confusion[:, idx].sum() - TP
        FN = confusion[idx, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        dice = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        metrics[f'cls{idx}'] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'dice': round(dice, 4),
            'iou': round(iou, 4)
        }
    
    # 日志输出
    log_info = f'test of best model, loss: {np.mean(loss_list):.4f}\n' \
               f'-> 平均IoU: {np.mean([v["iou"] for v in metrics.values()]):.4f}, ' \
               f'平均Dice: {np.mean([v["dice"] for v in metrics.values()]):.4f}\n' \
               f'-> 各类别指标:\n' + \
               '\n'.join([f'类别{idx}: Precision={v["precision"]}, Recall={v["recall"]}, Dice={v["dice"]}, IoU={v["iou"]}' 
                          for idx, v in metrics.items()]) + \
               f'\n混淆矩阵:\n{confusion}'
    
    print(log_info)
    logger.info(log_info)
    
    return np.mean(loss_list)
