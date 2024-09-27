from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
import argparse
from train import Solver
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from dataset import MyDataset
from torchvision.transforms.functional import InterpolationMode
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import autoaugment, RandomErasing
# de4f12de661c472b225922cced0320e47de3191c

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    autoaugment.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    ),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    ),
])

valid_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 224)
    transforms.Resize((128, 128)),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main(config):
    wandb.init(project="Hw1-1_downstream_New", config=config)
    # load the dataset
    train_set = MyDataset(
        root='D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/office/train', transform=train_tfm)
    valid_set = MyDataset(
        root='D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/office/val', transform=valid_tfm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model setting
    model = torchvision.models.resnet50(weights=None)

    checkpoint_path = 'D:/python/DLCV/HW1/p1/New/best_model_70_pretrained.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1000),
        # nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1000, 65)
    )
    model = model.to(device)

    # Data loader.
    trainset_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valset_loader = DataLoader(
        valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # optimizer
    opt_name = config.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=1e-4
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=1e-4)
    else:
        raise RuntimeError(
            f"Invalid optimizer {config.opt}. Only SGD, RMSprop and AdamW are supported.")
    # schdular
    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 10, T_mult=2
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=0
    )
    # training
    for ep in range(config.epoch):
        print(f'-----Epoch : {ep+1}-----')
        solver = Solver(model, trainset_loader, valset_loader,
                        ep, device, optimizer, lr_scheduler, config)
        train_acc, train_loss = solver.train_one_epoch()
        val_acc, val_loss = solver.evaluate()
        # lr_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        # output
        solver.save_checkpoint()
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc, "learning_rate": current_lr})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--ckp_dir', default='D:/python/DLCV/HW1/p1/New/result',
                        type=str, help='Checkpoint path')
    parser.add_argument('--name', default='G1', type=str,
                        help='Name for saving model')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='mini-batch size')
    parser.add_argument('--lr', type=int, default=1e-4, help='lr')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of total epoch')

    parser.add_argument('--resume_iter', type=int, default=0,
                        help='resume training from this iteration')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=900)
    parser.add_argument(
        "--data_aug", help="data augmentation", action="store_true")

    config = parser.parse_args()
    print(config)
    main(config)
