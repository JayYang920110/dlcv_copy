import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as trns
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import sys
# Dataset class for inference


class DatasetInfer(Dataset):
    def __init__(self, file_path, transform=None):
        self.path = file_path
        self.transform = transform or self.default_transform()
        self.files = self.load_image_files(self.path)

    def load_image_files(self, path):
        return sorted([x for x in os.listdir(path) if x.endswith(".jpg")])

    def default_transform(self):
        return trns.Compose([
            trns.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),
            trns.ToTensor(),
            trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.files[idx])
        img = self.load_image(img_path)
        img = self.transform(img)
        return img, self.files[idx]

    def load_image(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.files)


def load_model(device, checkpoint_path):
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1000),
        nn.Dropout(p=0.5),
        nn.Linear(1000, 65)
    )
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=torch.device('cpu'))['model_state_dict'])
    return model.to(device)


def perform_inference(model, device, val_loader):
    model.eval()
    predict_list, img_name_list = [], []

    with torch.inference_mode():
        for images, img_name in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            predict_list.extend(predictions.detach().cpu().tolist())
            img_name_list.extend(img_name)

    return np.array(img_name_list, dtype=str), np.array(predict_list, dtype=np.uint8)


def write_predictions_to_csv(pred_csv_path, np_img_name, np_predict):
    with open(pred_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "filename", "label"))
        for id, (img_name, predict) in enumerate(zip(np_img_name, np_predict)):
            writer.writerow([id, img_name, predict])


def main():
    print('Strating')
    # test_csv = "D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/office/val.csv"
    # img_dir_test = "D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/office/val"
    # test_pred_csv = "D:/python/DLCV/HW1/p1/result/result2.csv"

    test_csv = sys.argv[1]
    img_dir_test = sys.argv[2]
    test_pred_csv = sys.argv[3]
    checkpoint_path = './p1_finetune_weight.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_data = DatasetInfer(img_dir_test)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    model = load_model(device, checkpoint_path)

    np_img_name, np_predict = perform_inference(
        model, device, val_loader)

    write_predictions_to_csv(test_pred_csv, np_img_name, np_predict)
    print("Program all done.")


if __name__ == "__main__":
    main()
