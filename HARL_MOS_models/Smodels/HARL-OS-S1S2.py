import math
import os

import numpy as np
import scipy.io as sio

from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score, classification_report
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights, resnet34
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder, CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

bird_classes = [
    "AMCR", "AMRO", "BLJA",
    "ECMC", "ECMK", "FFCR",
]

# 路径定义
gamma_path = r"F:\s1s2\s1s2\gamma\DB2"
mel_path = r"F:\s1s2\s1s2\mel\DB2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

region_paths = {
    'region1': {'gamma': os.path.join(gamma_path, 'S1'),
                'mel': os.path.join(mel_path, 'S1')},
    'region2': {'gamma': os.path.join(gamma_path, 'S2'),
                'mel': os.path.join(mel_path, 'S2')},

}



class StyleRandomization(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1) > self.p:
            return x
        # x: [B, C, H, W]
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-6
        x = (x - mean) / std

        # 随机打乱 mean 和 std
        perm = torch.randperm(x.size(0))
        shuffled_mean = mean[perm]
        shuffled_std = std[perm]

        x = x * shuffled_std + shuffled_mean
        return x



def balance_dataset(data, labels):
    """
    使用 RandomOverSampler 进行数据过采样，使类别均衡
    """
    ros = RandomOverSampler(sampling_strategy="auto")  # 让所有类别数量均衡
    indices = np.arange(len(labels)).reshape(-1, 1)  # 创建索引，以便对数据进行采样
    resampled_indices, resampled_labels = ros.fit_resample(indices, labels)

    # 取出被过采样的数据
    resampled_data = [data[i] for i in resampled_indices.flatten()]


    return resampled_data, resampled_labels



class ResNetBranch(nn.Module):
    def __init__(self, input_channels=3, embed_dim=2048, num_heads=8, args=None):
        super(ResNetBranch, self).__init__()

        # ResNet Backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self._modify_first_conv(input_channels)
        self.resnet.fc = nn.Identity()

        # ✅ MixStyle 插入
        self.mixstyle = StyleRandomization(p=0.4)

        # ✅ 多头注意力
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def _modify_first_conv(self, input_channels):
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

    def forward(self, x):
        # ResNet Feature Extract
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.mixstyle(x)

        x = self.resnet.layer2(x)
        x = self.mixstyle(x)

        x = self.resnet.layer3(x)
        x = self.mixstyle(x)

        x = self.resnet.layer4(x)  # Shape: (B, 2048, H, W)

        # 平铺后送入注意力模块
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # Shape: (B, N, C), N = H * W

        # Multihead Attention
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        x = self.norm(attn_out + x)  # Add & Norm

        # 平均池化整合注意力信息
        x = x.mean(dim=1)  # (B, C)

        return x
# 双分支主模型（去掉 BatchNorm，添加 Transformer）


class MultiBranchCNN_G_M_WI(nn.Module):
    def __init__(self, num_classes,in_channels=3, num_sound_types=2, args=None):
        super(MultiBranchCNN_G_M_WI, self).__init__()




        self.gamma_branch = ResNetBranch(input_channels=in_channels, args=args)
        self.mel_branch = ResNetBranch(input_channels=in_channels, args=args)
        # 语音识别模型


        # 主任务分类器
        self.fc_main = nn.Sequential(
            nn.Linear(4096, 512),  # 现在有三个分支
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )



        self.brb_weight = nn.Parameter(torch.tensor(0.7))
    def forward(self, gamma, mel):
        # 处理 Gamma 和 Mel 数据
        gamma = gamma.repeat(1, 3, 1, 1)
        mel = mel.repeat(1, 3, 1, 1)
        gamma_features = self.gamma_branch(gamma)
        mel_features = self.mel_branch(mel)



        combined = torch.cat([gamma_features, mel_features], dim=1)

        return self.fc_main(combined)


def compute_deltas(features):
    delta = np.diff(features, axis=1, append=features[:, -1:])  # 时间方向差分
    delta_delta = np.diff(delta, axis=1, append=delta[:, -1:])  # 再次差分
    return delta, delta_delta

# 数据加载部分保持不变（需根据实际情况调整输入维度）
class BirdDataset(Dataset):
    def __init__(self, region_path, bird_classes, transform=None, augmentation=None):
        self.data = []
        self.labels = []  # 主任务标签（鸟类分类）

        self.transform = transform
        self.augmentation = augmentation

        for label, bird_class in enumerate(bird_classes):
            gamma_dir = os.path.join(region_path['gamma'], bird_class)
            mel_dir = os.path.join(region_path['mel'], bird_class)

            if not os.path.exists(gamma_dir) or not os.path.exists(mel_dir):
                continue

            for file in os.listdir(gamma_dir):
                if file.endswith('.mat'):
                    gamma_file = os.path.join(gamma_dir, file)
                    mel_file = os.path.join(mel_dir, file)

                    # 加载 Gamma 特征
                    gamma_data = sio.loadmat(gamma_file)
                    gamma_feature = gamma_data.get('gammaSpecdb')
                    if gamma_feature is None:
                        continue

                    gamma_delta, gamma_delta_delta = compute_deltas(gamma_feature)
                    gamma_combined = np.concatenate([gamma_feature, gamma_delta, gamma_delta_delta], axis=0)
                    gamma_combined = np.expand_dims(gamma_combined, axis=0)

                    # 加载 Mel 特征
                    mel_data = sio.loadmat(mel_file)
                    mel_feature = mel_data.get('melSpecdb')
                    if mel_feature is None:
                        continue

                    mel_delta, mel_delta_delta = compute_deltas(mel_feature)
                    mel_combined = np.concatenate([mel_feature, mel_delta, mel_delta_delta], axis=0)
                    mel_combined = np.expand_dims(mel_combined, axis=0)

                    # 保存数据
                    self.data.append((gamma_combined, mel_combined))
                    self.labels.append(label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gamma_feature, mel_feature = self.data[idx]
        label = self.labels[idx]


        # 应用增强
        if self.augmentation:
            gamma_feature = self.augmentation(gamma_feature)
            mel_feature = self.augmentation(mel_feature)

        # 应用变换
        if self.transform:
            gamma_feature = self.transform(gamma_feature)
            mel_feature = self.transform(mel_feature)

        return (gamma_feature, mel_feature), label

# 原有实现保持不变...
def load_data(region_paths, bird_classes):
    datasets = {region: BirdDataset(region_path=paths, bird_classes=bird_classes)
                for region, paths in region_paths.items()}
    data_loaders = {}


    train_dataset = datasets['region1']

    test_dataset_region3 = datasets[('region2')]
    train_dataset.data, train_dataset.labels = balance_dataset(train_dataset.data, train_dataset.labels)
    data_loaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    data_loaders['test_region2'] = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)

    return data_loaders


# 训练流程
def train_model(model, criterion_main, optimizer, train_loader, val_loader1,  num_epochs=50, save_epoch=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_model_wts = None
    best_avg_val_acc_1 = 0.0  # 记录两个验证集平均准确率的最佳值
    best_avg_val_acc_2 = 0.0  # 记录两个验证集平均准确率的最佳值

    def evaluate_on_loader(loader):
        model.eval()
        all_preds_main = []
        all_labels = []

        with torch.no_grad():
            for (gamma, mel), labels in loader:
                gamma = gamma.float().to(device)
                mel = mel.float().to(device)
                labels = labels.to(device)

                main_output= model(gamma, mel)
                _, preds_main = torch.max(main_output, 1)

                all_preds_main.append(preds_main.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds_main = np.concatenate(all_preds_main)
        all_labels = np.concatenate(all_labels)

        f1_main = f1_score(all_labels, all_preds_main, average='macro')
        return f1_main

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects_main = 0


        for (gamma, mel), labels in train_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)


            optimizer.zero_grad()
            main_output= model(gamma, mel)
            loss_main = criterion_main(main_output, labels)


            loss_main.backward()
            optimizer.step()

            _, preds_main = torch.max(main_output, 1)


            running_loss += loss_main.item() * gamma.size(0)
            running_corrects_main += torch.sum(preds_main == labels.data)


            del gamma, mel, labels,  main_output, preds_main
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_main = running_corrects_main.double() / len(train_loader.dataset)


        print(f'[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} | Main Acc: {epoch_acc_main:.4f} ')
        scheduler.step(epoch_loss)

        # 验证集 Region 1 和 Region 3
        val_acc1 = evaluate_on_loader(val_loader1)


        print(f'           >>> Val Acc1: {val_acc1:.4f} ')

        # 保存验证集准确率平均最高的模型
        if val_acc1 > best_avg_val_acc_1:

                best_avg_val_acc_1 = val_acc1

                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "Resnet_GM_OverS_Mixstyle_noMultitask_s1_best1_avg_val.pth")
                print(f"New best model saved at epoch {epoch + 1} with Avg Val Acc: {val_acc1:.4f}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

import seaborn as sns


def evaluate_model(model, test_loader, region_name=None):
    model.eval()
    all_preds_main = []
    all_preds_aux = []
    all_labels = []


    running_corrects_main = 0


    with torch.no_grad():
        for (gamma, mel), labels in test_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)


            main_output = model(gamma, mel)
            _, preds_main = torch.max(main_output, 1)


            all_preds_main.append(preds_main.cpu().numpy())

            all_labels.append(labels.cpu().numpy())


            running_corrects_main += torch.sum(preds_main == labels.data)


    torch.cuda.empty_cache()

    all_preds_main = np.concatenate(all_preds_main)

    all_labels = np.concatenate(all_labels)


    # 主任务评估
    main_accuracy = running_corrects_main.double() / len(test_loader.dataset)

    # 计算辅助任务评估指标


    # 计算混淆矩阵、召回率和F1分数（主任务和辅助任务）
    print(
        f"Confusion Matrix for Main Task (Bird Classification) in {region_name}:\n{confusion_matrix(all_labels, all_preds_main)}")


    recall_main = recall_score(all_labels, all_preds_main, average='macro')
    f1_main = f1_score(all_labels, all_preds_main, average='macro')


    print(
        f"Main Task (Bird Classification),Accuracy for {region_name}: {main_accuracy:.4f}, Recall: {recall_main:.4f}, F1 Score: {f1_main:.4f}")

    f1 = f1_score(all_labels, all_preds_main, average='macro')
    accuracy = running_corrects_main.double() / len(test_loader.dataset)
    print(f'Accuracy: {accuracy:.4f}, UAR: {recall_main:.4f}, F1 Score: {f1:.4f}')
    # 每个类别详细统计
    print(f"\nDetailed Classification Report for {region_name} (Main Task):")
    report = classification_report(all_labels, all_preds_main, target_names=bird_classes, digits=4)
    print(report)

    return f1
def get_model():
    return MultiBranchCNN_G_M_WI(num_classes=10,in_channels=3)
# 主程序
if __name__ == '__main__':
    # 初始化模型
    model = MultiBranchCNN_G_M_WI(num_classes=10).to(device)
    criterion_main = CrossEntropyLoss()


    # model.load_state_dict(torch.load('Resnet_GM_OverS_Mixstyle_attention_d3.pth'))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# bird_classes = [...]  # 填入实际的鸟类类别列表
    data_loaders = load_data(region_paths, bird_classes)  # 需要实现load_data函数
    # model = train_model(model, criterion_main, optimizer, data_loaders['train'],
    #                     data_loaders['test_region2'], num_epochs=50, save_epoch=50)
    model.eval()
    model.load_state_dict(
        torch.load("Resnet_GM_OverS_Mixstyle_noMultitask_s1_best1_avg_val.pth"))

    # 开始训练
    # train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'])

    evaluate_model(model, data_loaders['test_region2'], "Region 2")
