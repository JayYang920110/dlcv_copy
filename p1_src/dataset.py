import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as trns


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.path = root
        self.transform = transform
        if transform:
            self.transform = transform
        else:
            self.transform = trns.Compose([
                trns.Resize([128, 128]),
                trns.ToTensor(),
                trns.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

        self.files = sorted(
            [x for x in os.listdir(self.path) if x.endswith(".jpg")])

    def __getitem__(self, idx):

        img_path = os.path.join(self.path, self.files[idx])
        with open(img_path, "rb") as f:
            data = Image.open(f)
            data.convert("RGB")
        data = self.transform(data)

        # 獲取檔案名稱並從檔案名提取標籤
        imgname = self.files[idx]
        label = int(imgname.split('_')[0])  # 假設標籤與檔案名以'_'分隔

        return data, label  # 返回數據、標籤和檔案名

    def __len__(self):
        return len(self.files)
