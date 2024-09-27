import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from model import Model
from tqdm import tqdm
import time


class Solver(object):
    def __init__(self, model,  train_loader, valid_loader, epoch, device, optimizer, schedular, config):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.epoch = epoch
        self.iteration = config.resume_iter
        self.model = model
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        self.optimizer = optimizer
        self.schedular = schedular
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()  # 設置模型為訓練模式
        total_loss = 0
        correct = 0
        total = 0

        # tqdm 進度條
        progress = tqdm(self.train_loader, desc='Training', leave=True)
        for i, (image, target) in enumerate(progress):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            # 累計損失
            total_loss += loss.item()
            # 計算準確率
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 更新進度條
            progress.set_postfix(loss=loss.item(
            ), accuracy=100. * correct / total, time=time.time() - start_time)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return accuracy, avg_loss

    def evaluate(self):
        self.model.eval()  # 設置模型為評估模式
        total_loss = 0
        correct = 0
        total = 0

        with torch.inference_mode():  # 更高效的推理模式
            progress = tqdm(self.valid_loader, desc='Validation', leave=True)
            for i, (image, target) in enumerate(progress):
                start_time = time.time()
                image, target = image.to(self.device), target.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # 更新進度條顯示當前批次的損失和準確率
                progress.set_postfix(loss=loss.item(
                ), accuracy=100. * correct / total, time=time.time() - start_time)

            avg_loss = total_loss / len(self.valid_loader)  # 使用正確的分母計算平均損失
            accuracy = 100. * correct / total  # 計算準確率

            return accuracy, avg_loss

    def save_checkpoint(self):
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.schedular.state_dict(),
        }
        checkpoint_path = os.path.join(
            self.ckp_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(state, checkpoint_path)
