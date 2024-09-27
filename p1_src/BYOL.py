import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from byol_pytorch import BYOL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
# test model, a resnet 50


resnet = models.resnet50(weights=None)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, default='D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/mini/train',
                    help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-2
IMAGE_SIZE = 128
# weight_decay = 0.0001
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('train_loss', loss, on_epoch=True)
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=LR)
# , weight_decay=weight_decay
        # # Add the scheduler
        lr_scheduler = {
            # 'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True),
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2),
            'monitor': 'train_loss'  # This tells the scheduler which metric to monitor
        }

        # Return optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def save_resnet_weights(self, save_path):
        resnet_weights = model.learner.online_encoder.state_dict()
        torch.save(resnet_weights, save_path)
        print(f'ResNet50 weights saved to {save_path}')

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        save_path = f'D:/python/DLCV/HW1/hw1_data/checkpoints/resnet50_epoch_{epoch:02d}.pth'
        self.save_resnet_weights(save_path)

# images dataset


# def expand_greyscale(t):
#     return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        print(folder)
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
            # transforms.RandomRotation(10),       # 隨機旋轉（-10度到+10度）
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
            # transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


# main
if __name__ == '__main__':
    print(f'device is GPU : {torch.cuda.is_available()}')
    print(torch.version.cuda)
    wandb_logger = WandbLogger(project="Hw1-1")
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, persistent_workers=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='D:/python/DLCV/HW1/hw1_data/checkpoints/15',
        filename='{epoch:02d}-{train_loss:.2f}',
        verbose=True,
        monitor='train_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        logger=wandb_logger,  # 加入 WandB logger
        callbacks=[checkpoint_callback],  # 加入 checkpoint callback

    )
    trainer.fit(model, train_loader)
    # checkpoint_path = 'D:/python/DLCV/HW1/hw1_data/checkpoints/13/epoch=94-train_loss=0.04.ckpt'
    # trainer.fit(model, train_loader, ckpt_path=checkpoint_path)
