import datetime
import os
import random
from tqdm import tqdm
import numpy as np

from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
# import utils.joint_transforms as joint_transforms
# import utils.transforms as extended_transforms
# from VGG16FCN import FCN32VGG
# from VGG16FCN import FCN8s
# from VGG16FCN import UNet
from model import DeepLabv3
from Dataset import MyDataset
import wandb
from torch.optim.lr_scheduler import OneCycleLR

cudnn.benchmark = True

ckpt_path = 'D:\python\DLCV\HW1\p2\Train\checkpoints'
exp_name = 'DeepLabv-18,onecycle'
args = {
    'batch_size': 4,
    'epoch_num': 80,
    'lr': 3e-4,  # 1e-3
    # 'lr_schedular_factor': 0.1,
    # 'lr_scheular_patient': 5,  # 10
    # 'weight_decay': 1e-4,
    # 'input_size': (512, 512),
    # 'momentum': 0.95,
    # 'snapshot': '',  # empty string denotes no snapshot
    # 'print_freq': 20,
    # 'val_save_to_img_file': False,
    # 'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}


# def convert_to_tensor(x):
#     return torch.tensor(np.array(x, dtype=np.uint8), dtype=torch.long)
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        ignore_index=6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        log_pt = -self.CE(logits, labels)
        loss = -((1 - torch.exp(log_pt)) ** self.gamma) * self.alpha * log_pt
        return loss


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp + tp_fn - tp) == 0:
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f' % (i, iou))
    return mean_iou
    # print('\nmean_iou: %f\n' % mean_iou)


def train(train_loader, net, criterion, optimizer, epoch, scheduler):
    net.train()
    train_loss = 0
    all_preds = []
    all_gt = []
    # total_iou = 0
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f"Epoch {epoch}")

    for i, data in pbar:

        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        logits = outputs['out']
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds.append(preds)
        all_gt.append(labels_np)
        pbar.set_postfix({'loss': train_loss / (i + 1)})

    train_loss /= len(train_loader)
    mIoU = mean_iou_score(np.concatenate(
        all_preds, axis=0), np.concatenate(all_gt, axis=0))
    all_preds.clear()
    all_gt.clear()
    return train_loss, mIoU


def validate(val_loader, net, criterion):
    net.eval()

    with torch.inference_mode():
        val_loss = 0
        # total_iou = 0
        all_preds = []
        all_gt = []

        for vi, data in enumerate(val_loader):
            inputs, gts = data

            inputs = inputs.cuda()
            gts = gts.cuda()

            outputs = net(inputs)['out']
            predictions = outputs.argmax(dim=1)

            loss = criterion(outputs, gts)
            val_loss += loss.item()

            predictions = predictions.detach().cpu().numpy()
            gts = gts.detach().cpu().numpy()
            all_preds.append(predictions)
            all_gt.append(gts)
        mIoU = mean_iou_score(np.concatenate(
            all_preds, axis=0), np.concatenate(all_gt, axis=0))
        val_loss /= len(val_loader)
        all_preds.clear()
        all_gt.clear()

    net.train()
    return val_loss, mIoU


def main():

    wandb.init(project="Hw1-2")
    net = DeepLabv3(model='resnet50').cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # 归一化
    ])

    train_set = MyDataset(
        'D:/python/DLCV/HW1/hw1_data/hw1_data/p2_data/train', train=True, transform=transform, augmentation=True)
    train_loader = DataLoader(
        train_set, batch_size=args['batch_size'], num_workers=0, pin_memory=True, shuffle=True)

    val_dataset = MyDataset('D:/python/DLCV/HW1/hw1_data/hw1_data/p2_data/validation',
                            train=True, transform=transform)
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'], num_workers=0, pin_memory=True, shuffle=False)

    loss_fn = FocalLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args['lr'])
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=args['lr_schedular_factor'], patience=args['lr_scheular_patient'], verbose=True)
    steps_per_epoch = len(train_loader)  # 每个 epoch 有多少个 batch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,  # 设置学习率峰值
        steps_per_epoch=steps_per_epoch,
        epochs=args['epoch_num']
    )
# D:/python/DLCV/github1/dlcv-fall-2024-hw1-JayYang920110/problem2/checkpoint/best_model_epoch_18_mIoU_0.7363.pt
    checkpoint_file = ''
    if os.path.exists(checkpoint_file):
        print(f"加載檢查點 '{checkpoint_file}'")
        model_dict = net.state_dict()
        pretrain_dict = torch.load(checkpoint_file)
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if k in model_dict}
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 如果有調度器，初始化並加載它的狀態
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # start_epoch = checkpoint['epoch'] + 1  # 從儲存的下一個epoch開始訓練
        # print(f"從 epoch {start_epoch} 繼續訓練")
    else:
        print(f"檢查點文件不存在,從0開始訓練")
        start_epoch = 0
    net.train()

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    file_path = os.path.join(ckpt_path, exp_name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for epoch in range(args['epoch_num']):
        train_loss, train_mIoU = train(
            train_loader, net, loss_fn, optimizer, epoch, scheduler)
        val_loss, val_mIoU = validate(val_loader, net, loss_fn)

        # 保存每次训练的检查点
        if (epoch + 1) % 5 == 0 or epoch >= 50:
            checkpoint_name = f'epoch_{epoch}_val_mIoU_{val_mIoU:.5f}.pth'
            checkpoint_path = os.path.join(
                ckpt_path, exp_name, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch[{epoch}], Train-loss:{train_loss}, Train-mIoU:{train_mIoU}')
        print(f'               val-loss:{val_loss}, val-mIoU:{val_mIoU}')
        wandb.log({"train_loss": train_loss, "train_mIoU": train_mIoU,
                  "val_loss": val_loss, "val_mIoU": val_mIoU, "learning_rate": current_lr})


if __name__ == '__main__':
    main()
