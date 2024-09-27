import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from dataset import MyDataset
# 檢查 GPU 或 CPU 設置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型並移至設備
model = torchvision.models.resnet50(weights=None)


model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1000),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 65)
)
model.to(device)

# 加載你保存的權重文件
checkpoint_path = 'D:/python/DLCV/HW1/p1/New/result/G/checkpoint_epoch_196.pth'

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

model_weights = checkpoint['model_state_dict']

# 將權重加載到模型中
model.load_state_dict(model_weights)
# 鎖定參數，避免對其進行訓練
for param in model.parameters():
    param.requires_grad = False

# 設置模型到評估模式
model.eval()

# 定義鉤子來提取第二層的特徵
features = []


def hook(module, input, output):
    features.append(output.detach().cpu().numpy())


# 註冊鉤子到倒數第二層
handle = model.fc[0].register_forward_hook(hook)

# 定義第一層卷積層的鉤子
first_layer_features = []


def first_layer_hook(module, input, output):
    first_layer_features.append(output.detach().cpu().numpy())


# 註冊鉤子到第一層卷積層
first_layer_handle = model.conv1.register_forward_hook(first_layer_hook)

tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = MyDataset(
    root='D:/python/DLCV/HW1/hw1_data/hw1_data/p1_data/office/train', transform=tfm)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
)

# 用來存放 t-SNE 結果的列表
outputs, labels = [], []
first_layer_outputs = []

# 對訓練集進行推論
with torch.no_grad():
    for inputs, label in data_loader:
        inputs = inputs.to(device)
        model(inputs)  # 執行前向傳播
        outputs.append(features[0])  # 收集第二層的特徵

        labels.append(label.cpu().numpy())
        features = []  # 重置第二層特徵緩存


# 將輸出和標籤轉換為 numpy 格式
outputs = np.concatenate(outputs, axis=0)
labels = np.concatenate(labels, axis=0)

# 使用 t-SNE 降維第二層
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(outputs)

# 視覺化第二層的 t-SNE 結果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of the second-last layer')
plt.show()


# 移除鉤子
handle.remove()
