import math
import os
import random

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

song_birds = [
    "AMRO", "FFCR",
]
call_birds = [
    "ECMC", "AMCR", "BLJA",
    "ECMK"
]
# 路径定义
gamma_path = r"F:\s1s2\s1s2\gamma\DB2"
mel_path = r"F:\s1s2\s1s2\mel\DB2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子，确保实验可重复性
def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况下
    # 确保CUDA的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用随机种子设置函数
set_random_seed(seed=42)

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
    def __init__(self, num_classes, in_channels=3,num_sound_types=2, args=None):
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

        # 辅助任务分类器
        self.fc_aux = nn.Sequential(
            nn.Linear(4096, 512),  # 现在有三个分支

            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_sound_types)
        )


        self.brb_weight = nn.Parameter(torch.tensor(0.7))
    def forward(self, gamma, mel):
        # 处理 Gamma 和 Mel 数据
        gamma = gamma.repeat(1, 3, 1, 1)
        mel = mel.repeat(1, 3, 1, 1)
        gamma_features = self.gamma_branch(gamma)
        mel_features = self.mel_branch(mel)



        combined = torch.cat([gamma_features, mel_features], dim=1)

        return self.fc_main(combined), self.fc_aux(combined)


def compute_deltas(features):
    delta = np.diff(features, axis=1, append=features[:, -1:])  # 时间方向差分
    delta_delta = np.diff(delta, axis=1, append=delta[:, -1:])  # 再次差分
    return delta, delta_delta
def get_sound_type(bird_class):
    if bird_class in song_birds:
        return 1  # songs
    elif bird_class in call_birds:
        return 0  # calls
    return -1  # Invalid class (should not happen)
# 学习率预热调度器
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.scheduler = scheduler
        self.current_epoch = 0
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
    
    def step(self, metrics=None):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段，线性增加学习率
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup Epoch {self.current_epoch}/{self.warmup_epochs}, Learning Rate: {lr:.6f}")
        else:
            # 预热结束后使用主调度器
            if metrics is not None:
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()

# 数据加载部分保持不变（需根据实际情况调整输入维度）
class BirdDataset(Dataset):
    def __init__(self, region_path, bird_classes, transform=None, augmentation=None):
        self.data = []
        self.labels = []  # 主任务标签（鸟类分类）
        self.sound_types = []  # 辅助任务标签（声音类型分类）
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
                    self.sound_types.append(get_sound_type(bird_class))  # 获取声音类型标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gamma_feature, mel_feature = self.data[idx]
        label = self.labels[idx]
        sound_type = self.sound_types[idx]

        # 应用增强
        if self.augmentation:
            gamma_feature = self.augmentation(gamma_feature)
            mel_feature = self.augmentation(mel_feature)

        # 应用变换
        if self.transform:
            gamma_feature = self.transform(gamma_feature)
            mel_feature = self.transform(mel_feature)

        return (gamma_feature, mel_feature), (label, sound_type)

# 原有实现保持不变...
def load_data(region_paths, bird_classes):
    datasets = {region: BirdDataset(region_path=paths, bird_classes=bird_classes)
                for region, paths in region_paths.items()}
    data_loaders = {}


    train_dataset = datasets['region1']
    test_dataset_region2 = datasets['region2']


    data_loaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    data_loaders['test_region2'] = DataLoader(test_dataset_region2, batch_size=32, shuffle=False, drop_last=False)


    return data_loaders


# 训练流程
def train_model(model, criterion_main, criterion_aux, optimizer, train_loader, val_loader1, num_epochs=50, save_epoch=None, warmup_epochs=5):
    # 创建主学习率调度器 - ReduceLROnPlateau
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,  # 学习率衰减因子
        patience=5,   # 耐心值
        verbose=True,
        threshold=1e-4,  # 阈值，用于判断是否为改善
        threshold_mode='rel',  # 相对阈值
        min_lr=1e-7  # 最小学习率
    )
    
    # 创建预热调度器
    initial_lr = optimizer.param_groups[0]['lr'] * 0.1  # 预热初始学习率为目标学习率的1/10
    target_lr = optimizer.param_groups[0]['lr']
    scheduler = WarmupScheduler(optimizer, warmup_epochs, initial_lr, target_lr, main_scheduler)
    
    best_model_wts = None
    best_avg_val_acc_1 = 0.0  # 记录验证集准确率的最佳值

    def evaluate_on_loader(loader):
        model.eval()
        all_preds_main = []
        all_labels = []

        with torch.no_grad():
            for (gamma, mel), (labels, sound_types) in loader:
                gamma = gamma.float().to(device)
                mel = mel.float().to(device)
                labels = labels.to(device)

                main_output, _ = model(gamma, mel)
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
        running_corrects_aux = 0

        for (gamma, mel), (labels, sound_types) in train_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)
            sound_types = sound_types.to(device)

            optimizer.zero_grad()
            main_output, aux_output = model(gamma, mel)
            loss_main = criterion_main(main_output, labels)
            loss_aux = criterion_aux(aux_output, sound_types)
            loss = loss_main + loss_aux

            loss.backward()
            optimizer.step()

            _, preds_main = torch.max(main_output, 1)
            _, preds_aux = torch.max(aux_output, 1)

            running_loss += loss.item() * gamma.size(0)
            running_corrects_main += torch.sum(preds_main == labels.data)
            running_corrects_aux += torch.sum(preds_aux == sound_types.data)

            del gamma, mel, labels, sound_types, main_output, aux_output, preds_main, preds_aux
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_main = running_corrects_main.double() / len(train_loader.dataset)
        epoch_acc_aux = running_corrects_aux.double() / len(train_loader.dataset)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f'[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} | Main Acc: {epoch_acc_main:.4f} | Aux Acc: {epoch_acc_aux:.4f} | LR: {current_lr:.6f}')
        scheduler.step(epoch_loss)

        # 验证集 Region 1 和 Region 3
        val_acc1 = evaluate_on_loader(val_loader1)



        print(f'           >>> Val Acc1: {val_acc1:.4f} ')

        # 保存验证集准确率平均最高的模型
        if val_acc1 > best_avg_val_acc_1:

                best_avg_val_acc_1 = val_acc1

                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "Resnet_GM_Mixstyle_multitask_s1_best_avg_val.pth")
                print(f"New best model saved at epoch {epoch + 1} with Avg Val Acc: {val_acc1:.4f}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))  # 图像比例改为正方形
    sns.set(font_scale=1.2)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",

                cbar=False, square=True)  # 关键参数 square=True

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)  # 标签倾斜，防止重叠
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def evaluate_model(model, test_loader, region_name):
    model.eval()
    all_preds_main = []
    all_preds_aux = []
    all_labels = []
    all_sound_types = []

    running_corrects_main = 0
    running_corrects_aux = 0

    with torch.no_grad():
        for (gamma, mel), (labels, sound_types) in test_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)
            sound_types = sound_types.to(device)

            main_output, aux_output = model(gamma, mel)
            _, preds_main = torch.max(main_output, 1)
            _, preds_aux = torch.max(aux_output, 1)

            all_preds_main.append(preds_main.cpu().numpy())
            all_preds_aux.append(preds_aux.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_sound_types.append(sound_types.cpu().numpy())

            running_corrects_main += torch.sum(preds_main == labels.data)
            running_corrects_aux += torch.sum(preds_aux == sound_types.data)

    torch.cuda.empty_cache()

    all_preds_main = np.concatenate(all_preds_main)
    all_preds_aux = np.concatenate(all_preds_aux)
    all_labels = np.concatenate(all_labels)
    all_sound_types = np.concatenate(all_sound_types)

    # 主任务评估
    main_accuracy = running_corrects_main.double() / len(test_loader.dataset)

    # 计算辅助任务评估指标
    aux_accuracy = running_corrects_aux.double() / len(test_loader.dataset)

    # 计算混淆矩阵、召回率和F1分数（主任务和辅助任务）
    print(
        f"Confusion Matrix for Main Task (Bird Classification) in {region_name}:\n{confusion_matrix(all_labels, all_preds_main)}")

    print(
        f"Confusion Matrix for Auxiliary Task (Sound Type) in {region_name}:\n{confusion_matrix(all_sound_types, all_preds_aux)}")

    recall_main = recall_score(all_labels, all_preds_main, average='macro')
    f1_main = f1_score(all_labels, all_preds_main, average='macro')
    recall_aux = recall_score(all_sound_types, all_preds_aux, average='macro')
    f1_aux = f1_score(all_sound_types, all_preds_aux, average='macro')

    print(
        f"Main Task (Bird Classification),Accuracy for {region_name}: {main_accuracy:.4f}, Recall: {recall_main:.4f}, F1 Score: {f1_main:.4f}")
    print(
        f"Auxiliary Task (Sound Type) ,Accuracy (Sound Type) for {region_name}: {aux_accuracy:.4f},Recall: {recall_aux:.4f}, F1 Score: {f1_aux:.4f}")
    # 计算每个鸟类的Accuracy、Recall和F1 Score
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
    criterion_aux = CrossEntropyLoss()
    
    # 学习率调优配置
    # 使用AdamW优化器，初始学习率设置为5e-5（比原来的1e-4小，更稳定）
    # 权重衰减增加到1e-4，有助于正则化
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-5,          # 降低初始学习率以提高稳定性
        betas=(0.9, 0.999),  # Adam默认参数
        weight_decay=1e-4,    # 增加权重衰减以减少过拟合
        eps=1e-8           # 数值稳定性参数
    )
    
    # 打印学习率配置信息
    print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}")
    print(f"Weight Decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Warmup Epochs: 5")
#
#
# # bird_classes = [...]  # 填入实际的鸟类类别列表
    data_loaders = load_data(region_paths, bird_classes)  # 需要实现load_data函数
    # model = train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'],
    #                      data_loaders['test_region2'], num_epochs=50, save_epoch=50)
    model.eval()
    model.load_state_dict(
        torch.load("Resnet_GM_Mixstyle_multitask_s1_best_avg_val.pth"))
#
#     # 开始训练
#     # train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'])
    evaluate_model(model, data_loaders['test_region2'], "Region 2")
