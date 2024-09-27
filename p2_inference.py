import os
import sys
from torch.utils.data import Dataset
import imageio
import numpy as np
import torch
from torchvision import transforms
import glob
from PIL import Image
from copy import deepcopy
from torch import nn
from torchvision import models


class MyDataset(Dataset):
    def __init__(self, path, transform, train=False) -> None:
        super().__init__()

        self.Train = train
        self.Transform = transform
        self.Image_names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.Train:
            self.Mask_names = sorted(glob.glob(os.path.join(path, "*.png")))

    def __getitem__(self, idx):
        if self.Train:
            img = np.array(Image.open(self.Image_names[idx]))
            mask = np.array(Image.open(self.Mask_names[idx]))

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

            img = self.Transform(Image.fromarray(img))  # 转换回 PIL 之后应用变换
            mask = torch.tensor(mask, dtype=torch.long)  # 掩膜转为张量，类型为 long

            return img, mask
        else:
            img = Image.open(self.Image_names[idx])
            img = self.Transform(img)

            return img, os.path.basename(self.Image_names[idx])

    def __len__(self):
        return len(self.Image_names)


def DeepLabv3(outputchannels=7, model='resnet50'):
    if model == 'resnet50':
        print('Using Resnet50 backbone')
        model = models.segmentation.deeplabv3_resnet50(
            pretrained=True, progress=True)
    elif model == 'resnet101':
        print('Using Resnet101 backbone')
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)
    else:
        raise ValueError(
            f'Invalid model type {model}. Choose either "resnet50" or "resnet101".')
    model.classifier[4] = nn.Sequential(
        nn.Dropout2d(p=0.5),
        nn.Conv2d(256, outputchannels, 1, 1)
    )
    return model


def pred2image(batch_preds, batch_names, out_path):
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        name = name.replace('_sat', '_mask').replace('.jpg', '.png')
        imageio.imwrite(os.path.join(
            out_path, name.replace('.jpg', '.png')), pred_img)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


net = DeepLabv3(model='resnet50')
checkpoint = torch.load('./p2_weight.pth')
model_state_dict = checkpoint['model_state_dict']
net.load_state_dict(model_state_dict)
net = net.to(device)
net.eval()

input_folder = sys.argv[1]
output_folder = sys.argv[2]
# input_folder = 'D:/python/DLCV/HW1/hw1_data/hw1_data/p2_data/validation'
# output_folder = 'D:/python/DLCV/HW1/p2/result'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_dataset = MyDataset(
    input_folder,
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    train=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=0)

try:
    os.makedirs(output_folder, exist_ok=True)
except:
    pass
for x, filenames in test_loader:
    with torch.no_grad():
        x = x.to(device)
        out = net(x)['out']
    pred = out.argmax(dim=1)
    pred2image(pred, filenames, output_folder)
