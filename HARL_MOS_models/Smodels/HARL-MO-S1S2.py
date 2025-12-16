import math
import os
import random

import numpy as np
import scipy.io as sio

from imblearn.over_sampling import RandomOverSampler
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score, classification_report
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights, resnet34
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder, CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# 设置随机种子以确保实验可重复性
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用随机种子设置函数
set_random_seed(seed=42)

# 内存优化配置
opt_level = 'O1'  # mixed precision optimization level

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

region_paths = {
    'region1': {'gamma': os.path.join(gamma_path, 'S1'),
                'mel': os.path.join(mel_path, 'S1')},
    'region2': {'gamma': os.path.join(gamma_path, 'S2'),
                'mel': os.path.join(mel_path, 'S2')},

}



def balance_dataset(data, labels, sound_types):
    """
    使用 RandomOverSampler 进行数据过采样，使类别均衡
    """
    ros = RandomOverSampler(sampling_strategy="auto")  # 让所有类别数量均衡
    indices = np.arange(len(labels)).reshape(-1, 1)  # 创建索引，以便对数据进行采样
    resampled_indices, resampled_labels = ros.fit_resample(indices, labels)

    # 取出被过采样的数据
    resampled_data = [data[i] for i in resampled_indices.flatten()]
    resampled_sound_types = [sound_types[i] for i in resampled_indices.flatten()]

    return resampled_data, resampled_labels, resampled_sound_types



class ResNetBranch(nn.Module):
    def __init__(self, input_channels=3, embed_dim=2048, num_heads=8, args=None):
        super(ResNetBranch, self).__init__()

        # ResNet Backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self._modify_first_conv(input_channels)
        self.resnet.fc = nn.Identity()

        # ✅ MixStyle 插入


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


        x = self.resnet.layer2(x)


        x = self.resnet.layer3(x)


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
                    gamma_feature = gamma_feature.astype(np.float32)
                    gamma_delta = gamma_delta.astype(np.float32)
                    gamma_delta_delta = gamma_delta_delta.astype(np.float32)
                    gamma_combined = np.concatenate([gamma_feature, gamma_delta, gamma_delta_delta], axis=0)
                    gamma_combined = np.expand_dims(gamma_combined, axis=0)

                    # 加载 Mel 特征
                    mel_data = sio.loadmat(mel_file)
                    mel_feature = mel_data.get('melSpecdb')
                    if mel_feature is None:
                        continue

                    mel_delta, mel_delta_delta = compute_deltas(mel_feature)
                    mel_feature = mel_feature.astype(np.float32)
                    mel_delta = mel_delta.astype(np.float32)
                    mel_delta_delta = mel_delta_delta.astype(np.float32)
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
    test_dataset_region1 = datasets['region2']

    train_dataset.data, train_dataset.labels, train_dataset.sound_types = balance_dataset(train_dataset.data, train_dataset.labels,
                                                                           train_dataset.sound_types)
    # 减小批量大小以减少内存使用
    data_loaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    data_loaders['test_region2'] = DataLoader(test_dataset_region1, batch_size=32, shuffle=False, drop_last=False, pin_memory=True)


    return data_loaders


# 训练流程
def train_model(model, criterion_main, criterion_aux, optimizer, train_loader, val_loader1, num_epochs=50, save_epoch=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_model_wts = None
    best_avg_val_acc_1 = 0.0  # 记录验证集准确率的最佳值

    def evaluate_on_loader(loader):
        model.eval()
        all_preds_main = []
        all_labels = []

        with torch.no_grad():
            for i, ((gamma, mel), (labels, sound_types)) in enumerate(loader):
                # 移至设备
                gamma = gamma.float().to(device, non_blocking=True)
                mel = mel.float().to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # 前向传播
                main_output, _ = model(gamma, mel)
                _, preds_main = torch.max(main_output, 1)

                # 移动到CPU并转换为numpy
                all_preds_main.append(preds_main.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

                # 清理内存
                del gamma, mel, labels, sound_types, main_output, preds_main
                torch.cuda.empty_cache()

                # 每10个批次清理一次内存
                if i % 10 == 0:
                    torch.cuda.empty_cache()

        # 拼接结果
        all_preds_main = np.concatenate(all_preds_main)
        all_labels = np.concatenate(all_labels)

        f1_main = f1_score(all_labels, all_preds_main, average='macro')
        return f1_main

    # 尝试使用混合精度训练
    try:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        use_amp = True
        print("使用混合精度训练")
    except ImportError:
        use_amp = False
        print("混合精度训练不可用，使用常规训练")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects_main = 0
        running_corrects_aux = 0

        for i, ((gamma, mel), (labels, sound_types)) in enumerate(train_loader):
            # 移至设备
            gamma = gamma.float().to(device, non_blocking=True)
            mel = mel.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sound_types = sound_types.to(device, non_blocking=True)

            # 梯度清零
            optimizer.zero_grad(set_to_none=True)  # 更彻底的梯度清零

            # 使用混合精度训练
            if use_amp:
                with autocast():
                    main_output, aux_output = model(gamma, mel)
                    loss_main = criterion_main(main_output, labels)
                    loss_aux = criterion_aux(aux_output, sound_types)
                    loss = loss_main + loss_aux
                
                # 缩放梯度并反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                main_output, aux_output = model(gamma, mel)
                loss_main = criterion_main(main_output, labels)
                loss_aux = criterion_aux(aux_output, sound_types)
                loss = loss_main + loss_aux
                loss.backward()
                optimizer.step()

            # 计算准确率
            _, preds_main = torch.max(main_output, 1)
            _, preds_aux = torch.max(aux_output, 1)

            # 累加统计
            running_loss += loss.item() * gamma.size(0)
            running_corrects_main += torch.sum(preds_main == labels.data)
            running_corrects_aux += torch.sum(preds_aux == sound_types.data)

            # 清理内存
            del gamma, mel, labels, sound_types, main_output, aux_output, preds_main, preds_aux, loss, loss_main, loss_aux
            torch.cuda.empty_cache()

            # 每5个批次额外清理一次内存
            if i % 5 == 0:
                torch.cuda.empty_cache()

        # 计算epoch统计信息
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_main = running_corrects_main.double() / len(train_loader.dataset)
        epoch_acc_aux = running_corrects_aux.double() / len(train_loader.dataset)

        print(f'[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} | Main Acc: {epoch_acc_main:.4f} | Aux Acc: {epoch_acc_aux:.4f}')
        scheduler.step(epoch_loss)

        # 验证
        val_acc1 = evaluate_on_loader(val_loader1)
        print(f'           >>> Val Acc1: {val_acc1:.4f} ')

        # 保存最佳模型
        if val_acc1 > best_avg_val_acc_1:
            best_avg_val_acc_1 = val_acc1
            # 保存前清理内存
            torch.cuda.empty_cache()
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "Resnet_GM_OverS_s1_best_avg_val_111.pth")
            print(f"New best model saved at epoch {epoch + 1} with Val Acc: {val_acc1:.4f}")

    # 加载最佳模型
    if best_model_wts is not None:
        torch.cuda.empty_cache()
        model.load_state_dict(best_model_wts)

    return model




def evaluate_model(model, test_loader, region_name=None):
    model.eval()
    all_preds_main = []
    all_preds_aux = []
    all_labels = []
    all_sound_types = []

    running_corrects_main = 0
    running_corrects_aux = 0

    with torch.no_grad():
        for i, ((gamma, mel), (labels, sound_types)) in enumerate(test_loader):
            # 移至设备，使用non_blocking加速
            gamma = gamma.float().to(device, non_blocking=True)
            mel = mel.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sound_types = sound_types.to(device, non_blocking=True)

            # 前向传播
            main_output, aux_output = model(gamma, mel)
            _, preds_main = torch.max(main_output, 1)
            _, preds_aux = torch.max(aux_output, 1)

            # 移动到CPU并转换为numpy
            all_preds_main.append(preds_main.detach().cpu().numpy())
            all_preds_aux.append(preds_aux.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_sound_types.append(sound_types.detach().cpu().numpy())

            # 计算正确预测数
            running_corrects_main += torch.sum(preds_main == labels.data)
            running_corrects_aux += torch.sum(preds_aux == sound_types.data)

            # 清理内存
            del gamma, mel, labels, sound_types, main_output, aux_output, preds_main, preds_aux
            
            # 每5个批次清理一次内存
            if i % 5 == 0:
                torch.cuda.empty_cache()

    # 最终清理
    torch.cuda.empty_cache()

    # 拼接结果
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

if __name__ == '__main__':
    # 初始化模型
    model = MultiBranchCNN_G_M_WI(num_classes=6).to(device)
    criterion_main = CrossEntropyLoss()
    criterion_aux =  CrossEntropyLoss()

    # model.load_state_dict(torch.load('Resnet_GM_OverS_Mixstyle_attention_d3.pth'))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# bird_classes = [...]  # 填入实际的鸟类类别列表
    data_loaders = load_data(region_paths, bird_classes)  # 需要实现load_data函数
    # model = train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'],
    #                      data_loaders['test_region2'], num_epochs=50, save_epoch=50)
    model.eval()
    model.load_state_dict(
        torch.load("Resnet_GM_OverS_s1_best_avg_val_111.pth"))

    # 开始训练
    # train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'])
    evaluate_model(model, data_loaders['test_region2'], "Region 2")
