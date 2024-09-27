import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from torchvision.transforms.functional import hflip, vflip
from copy import deepcopy
import glob
import albumentations as A


class MyDataset(Dataset):
    def __init__(self, path, transform, train=False, augmentation=False) -> None:
        super().__init__()

        self.Train = train
        self.Transform = transform
        self.Image_names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.Train:
            self.Mask_names = sorted(glob.glob(os.path.join(path, "*.png")))

        if augmentation:
            print(f'Using data augmentation')
            # self.augmentation = A.Compose([    11 12
            #     A.HorizontalFlip(p=0.5),    # 水平翻转
            #     A.VerticalFlip(p=0.5),      # 垂直翻转
            #     A.Rotate(limit=90, p=0.5),  # 随机旋转（-30到30度）
            #     A.RandomResizedCrop(height=512, width=512, scale=(
            #         0.5, 1.0), p=0.5),  # 随机裁剪一部分并放大
            #     A.ColorJitter(brightness=0.2, contrast=0.2,
            #                   saturation=0.2, hue=0.2, p=0.5),
            # ], additional_targets={'mask': 'mask'})  # 指定掩膜为需要与图像同步的目标
            self.augmentation = A.Compose([  # 13 14

                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=0),
                # 隨機裁剪並縮放圖像到 224x224 尺寸，裁剪比例在 80% 到 100% 之間
                A.RandomResizedCrop(height=512, width=512,
                                    scale=(0.5, 1.0), p=0.5),

                # 隨機仿射變換，包含平移、旋轉、縮放和剪切
                # A.Affine(scale=(0.8, 1.2), rotate=(-30, 30),
                #          shear=(-10, 10), translate_percent=(0.1, 0.1), p=0.6),

                # 隨機透視變換，模擬透視效果變形
                # A.Perspective(scale=0.5, p=0.5),

                # 隨機抹去一部分圖像，模擬遮擋或噪聲，最多抹除32x32像素區域
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32,
                                min_holes=1, min_height=8, min_width=8, p=0.5),

            ], additional_targets={'mask': 'mask'})
            # self.augmentation = A.Compose([   #15
            #     A.HorizontalFlip(p=0.5),
            #     A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT,
            #              value=0, mask_value=0),
            #     A.RandomResizedCrop(height=512, width=512,
            #                         scale=(0.5, 1.0), p=0.5),
            #     A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), shear=(-10, 10), translate_percent=(0.1, 0.1),
            #              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
            #     # 对图像和掩码同时应用的安全变换
            #     # 以下变换仅应用于图像
            #     # A.OneOf([
            #     #     A.RandomBrightnessContrast(
            #     #         brightness_limit=0.2, contrast_limit=0.2, p=1),
            #     #     A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            #     #     A.Blur(blur_limit=7, p=1),
            #     #     A.HueSaturationValue(
            #     #         hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            #     # ], p=0.5)
            # ], additional_targets={'mask': 'mask'})

        else:
            self.augmentation = None

    def __getitem__(self, idx):
        if self.Train:
            # 打开图像和掩膜
            # 将图像转换为 NumPy 数组
            img = np.array(Image.open(self.Image_names[idx]))
            mask = np.array(Image.open(self.Mask_names[idx]))

            # 掩膜预处理
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            raw_mask = deepcopy(mask)

            mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
            mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
            mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
            mask[raw_mask == 2] = 3  # (Green: 010) Forest land
            mask[raw_mask == 1] = 4  # (Blue: 001) Water
            mask[raw_mask == 7] = 5  # (White: 111) Barren land
            mask[raw_mask == 0] = 6  # (Black: 000) Unknown

            # 应用数据增强（如果启用）
            if self.augmentation:

                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            # 将图像转换为 PyTorch 张量
            img = self.Transform(Image.fromarray(img))  # 转换回 PIL 之后应用变换
            mask = torch.tensor(mask, dtype=torch.long)  # 掩膜转为张量，类型为 long

            return img, mask
        else:
            img = Image.open(self.Image_names[idx])
            img = self.Transform(img)

            return img, os.path.basename(self.Image_names[idx])

    def __len__(self):
        return len(self.Image_names)


if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    # 設定資料路徑
    dir = 'D:/python/DLCV/HW1/hw1_data/hw1_data/p2_data/train'

    # 創建資料集實例
    dataset = MyDataset(dir, train=True)

    # 創建 DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    for images, masks in data_loader:
        print('Image shape:', images.shape)
        print('Mask shape:', masks.shape)
        mask_np = masks[0].numpy()
        unique_values = np.unique(mask_np)
        print('Unique values in mask:', unique_values)
        print('Number of unique values in mask:', len(unique_values))
        break
