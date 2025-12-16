import math
import os

import numpy as np
import scipy.io as sio

from imblearn.over_sampling import RandomOverSampler
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score, classification_report
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights, resnet34
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder, CrossEntropyLoss
import random
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

region_paths = {
    'region1': {'gamma': os.path.join(gamma_path, 'S1'),
                'mel': os.path.join(mel_path, 'S1')},
    'region2': {'gamma': os.path.join(gamma_path, 'S2'),
                'mel': os.path.join(mel_path, 'S2')},
}


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
        # x = self.mixstyle(x)
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
    def __init__(self, num_classes, in_channels=3, num_sound_types=2, args=None):
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

    test_dataset_region3 = datasets['region2']
    train_dataset.data, train_dataset.labels, train_dataset.sound_types = balance_dataset(train_dataset.data,
                                                                                          train_dataset.labels,
                                                                                          train_dataset.sound_types)
    data_loaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    data_loaders['test_region2'] = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)

    return data_loaders


# 训练流程
def train_model(model, criterion_main, criterion_aux, optimizer, train_loader, val_loader1, num_epochs=50,
                save_epoch=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_model_wts = None
    best_avg_val_acc_1 = 0.0  # 记录两个验证集平均准确率的最佳值

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

        print(
            f'[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} | Main Acc: {epoch_acc_main:.4f} | Aux Acc: {epoch_acc_aux:.4f}')
        scheduler.step(epoch_loss)

        # 验证集 Region 1 和 Region 3
        val_acc1 = evaluate_on_loader(val_loader1)

        print(f'           >>> Val Acc1: {val_acc1:.4f}')

        # 保存验证集准确率平均最高的模型
        if val_acc1 > best_avg_val_acc_1:
            best_avg_val_acc_1 = val_acc1

            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "Resnet_GM_OverS_Mixstyle123_mutitask_s1_best_avg_vaL_222.pth")
            print(f"New best model saved at epoch {epoch + 1} with Avg Val Acc: {val_acc1:.4f}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, recall_score, f1_score, classification_report

import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, recall_score, f1_score, classification_report


def evaluate_model(model, test_loader, region_name=None):
    model.eval()
    all_preds_main = []
    all_preds_aux = []
    all_labels = []
    all_sound_types = []

    running_corrects_main = 0
    running_corrects_aux = 0

    # 时间统计
    start_time = time.time()
    print(f"🕐 开始评估模型 - 区域: {region_name}")
    print(f"📅 评估开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 测试数据集大小: {len(test_loader.dataset)}")
    print(f"🔢 批次数量: {len(test_loader)}")
    print("-" * 60)

    # 存储每批次时间统计
    batch_times = []
    batch_start_times = []

    with torch.no_grad():
        for batch_idx, ((gamma, mel), (labels, sound_types)) in enumerate(test_loader):
            # 记录批次开始时间
            batch_start_time = time.time()
            batch_start_times.append(batch_start_time)

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

            # 记录批次结束时间并计算耗时
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            # 实时显示批次进度和时间
            print(f"Batch {batch_idx + 1:3d}/{len(test_loader)} | "
                  f"耗时: {batch_time:.4f}s | "
                  f"累计: {(batch_end_time - start_time):.2f}s | "
                  f"进度: {(batch_idx + 1) / len(test_loader) * 100:.1f}%")

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

    f1 = f1_score(all_labels, all_preds_main, average='macro')
    accuracy = running_corrects_main.double() / len(test_loader.dataset)

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time

    print(f'Accuracy: {accuracy:.4f}, UAR: {recall_main:.4f}, F1 Score: {f1:.4f}')

    # 添加时间戳信息
    print(f"\n⏰ 时间统计:")
    print(f"📅 评估开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 评估结束时间: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总评估耗时: {total_time:.4f} 秒")
    print(f"📅 评估日期: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')}")
    print(f"🕒 评估时间戳: {datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')}")

    # 添加批次时间统计
    print(f"\n📊 批次时间统计:")
    print(f"平均每批次耗时: {np.mean(batch_times):.4f} 秒")
    print(f"最快批次耗时: {np.min(batch_times):.4f} 秒")
    print(f"最慢批次耗时: {np.max(batch_times):.4f} 秒")
    print(f"批次耗时标准差: {np.std(batch_times):.4f} 秒")
    print(f"总批次数: {len(batch_times)}")

    # 显示前几个批次的详细时间（避免输出过多）
    print(f"\n🔍 前5个批次详细时间:")
    for i in range(min(5, len(batch_times))):
        batch_num = i + 1
        start_str = datetime.fromtimestamp(batch_start_times[i]).strftime('%H:%M:%S.%f')[:-3]
        print(f"  批次 {batch_num:2d}: {batch_times[i]:.4f}s (开始: {start_str})")

    if len(batch_times) > 5:
        print(f"  ... (省略 {len(batch_times) - 5} 个批次)")
        print(f"  最后1个批次: {batch_times[-1]:.4f}s")

    # 每个类别详细统计
    print(f"\nDetailed Classification Report for {region_name} (Main Task):")
    report = classification_report(all_labels, all_preds_main, target_names=bird_classes, digits=4)
    print(report)

    # 生成带时间戳的评估报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{region_name}_{timestamp}.txt"

    # 保存详细报告到文件
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("模型评估报告\n")
        f.write("=" * 50 + "\n")  # 修复：添加了等号
        f.write(f"评估区域: {region_name}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.4f} 秒\n")
        f.write(f"数据集大小: {len(test_loader.dataset)}\n")
        f.write(f"批次数量: {len(test_loader)}\n\n")

        # 写入批次时间统计
        f.write("批次时间统计:\n")
        f.write(f"平均每批次耗时: {np.mean(batch_times):.4f} 秒\n")
        f.write(f"最快批次耗时: {np.min(batch_times):.4f} 秒\n")
        f.write(f"最慢批次耗时: {np.max(batch_times):.4f} 秒\n")
        f.write(f"批次耗时标准差: {np.std(batch_times):.4f} 秒\n")
        f.write(f"总批次数: {len(batch_times)}\n\n")

        f.write("性能指标:\n")
        f.write(f"主任务准确率: {main_accuracy:.4f}\n")
        f.write(f"主任务召回率: {recall_main:.4f}\n")
        f.write(f"主任务F1分数: {f1_main:.4f}\n")
        f.write(f"辅助任务准确率: {aux_accuracy:.4f}\n")
        f.write(f"辅助任务召回率: {recall_aux:.4f}\n")
        f.write(f"辅助任务F1分数: {f1_aux:.4f}\n")
        f.write(f"总体准确率: {accuracy:.4f}\n")
        f.write(f"总体F1分数: {f1:.4f}\n")

        f.write("\n混淆矩阵:\n")
        f.write("主任务:\n")
        f.write(str(confusion_matrix(all_labels, all_preds_main)))
        f.write("\n\n辅助任务:\n")
        f.write(str(confusion_matrix(all_sound_types, all_preds_aux)))

        f.write(f"\n\n详细分类报告:\n")
        f.write(report)

    print(f"📄 评估报告已保存至: {report_filename}")

    return f1

def get_model():
    return MultiBranchCNN_G_M_WI(num_classes=6, in_channels=3)


if __name__ == '__main__':
    # 初始化模型
    model = MultiBranchCNN_G_M_WI(num_classes=10).to(device)
    criterion_main = CrossEntropyLoss()
    criterion_aux = CrossEntropyLoss()

    # model.load_state_dict(torch.load('Resnet_GM_OverS_Mixstyle_attention_d3.pth'))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # bird_classes = [...]  # 填入实际的鸟类类别列表
    data_loaders = load_data(region_paths, bird_classes)  # 需要实现load_data函数
    # model = train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'],
    #                     data_loaders['test_region2'], num_epochs=35, save_epoch=35)
    model.eval()
    model.load_state_dict(
        torch.load("Resnet_GM_OverS_Mixstyle123_mutitask_s1_best_avg_val_p=0.4.pth"))

    # 开始训练
    # train_model(model, criterion_main, criterion_aux, optimizer, data_loaders['train'])

    evaluate_model(model, data_loaders['test_region2'], "Region 2")
