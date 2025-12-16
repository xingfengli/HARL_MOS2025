import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import umap
from sklearn.preprocessing import LabelEncoder

from HARL_MOS_D3D2 import MultiBranchCNN_G_M_WI, BirdDataset, region_paths, bird_classes,DataLoader


# 修复后的特征提取函数
def extract_features(model, data_loader, device):
    """
    从数据加载器中提取特征 - 修复版本
    """
    model.eval()
    features = []  # 用于存放每个 batch 的特征 (numpy)
    labels = []  # ✅ 用于存放每个 batch 的标签 (numpy)
    sound_types = []  # 可选，用于存放声音类型
    batch_count = 0

    print(f"开始特征提取...")
    print(f"数据加载器长度: {len(data_loader)}")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            try:
                print(f"处理批次 {batch_idx + 1}/{len(data_loader)}")

                # 根据BirdDataset的返回格式处理数据
                # BirdDataset返回: ((gamma, mel), label, filename, bird_class, region)
                if len(batch_data) == 5:
                    (gamma, mel), batch_labels, filenames, bird_classes_batch, regions = batch_data
                # elif len(batch_data) == 2:
                #     (gamma, mel), batch_labels = batch_data
                elif len(batch_data) == 2:  # 可能包含sound_types
                    (gamma, mel), (batch_labels, sound_type_batch) = batch_data
                else:
                    print(f"警告: 未知的数据格式，批次 {batch_idx} 有 {len(batch_data)} 个元素")
                    continue

                # 检查数据是否为空
                if gamma is None or mel is None or batch_labels is None:
                    print(f"警告: 批次 {batch_idx} 数据为空")
                    continue

                # 检查数据形状
                print(f"  Gamma形状: {gamma.shape}, Mel形状: {mel.shape}, 标签形状: {batch_labels.to}")
                print(f"  当前批次标签唯一值: {np.unique(batch_labels)}")
                # 移动到设备
                if not isinstance(gamma, torch.Tensor):
                    gamma = torch.from_numpy(gamma) if isinstance(gamma, np.ndarray) else torch.tensor(gamma)
                gamma = gamma.float().to(device)

                if not isinstance(mel, torch.Tensor):
                    mel = torch.from_numpy(mel) if isinstance(mel, np.ndarray) else torch.tensor(mel)
                mel = mel.float().to(device)

                if not isinstance(batch_labels, torch.Tensor):
                    batch_labels = torch.tensor(batch_labels)
                batch_labels = batch_labels.to(device)
                # 模型推理
                main_output, aux_output = model(gamma, mel)  # ✅ 正确解包 tuple



                print(f"  模型输出形状: {aux_output.shape}")

                # 移动到CPU并转换为numpy
                features_np = main_output.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()
                # 打印当前批次的标签唯一值

                features.append(features_np)
                labels.append(labels_np)  # ✅ labels 是 list

                # # 如果有sound_types数据，也保存下来
                # if 'sound_type_batch' in locals():
                #     sound_types.append(sound_type_batch.cpu().numpy())

                batch_count += 1
                print(f"  成功处理批次 {batch_idx + 1}, 特征形状: {features_np.shape}")

            except Exception as e:
                print(f"处理批次{batch_idx}时出错: {e}")
                continue

    print(f"特征提取完成，共处理 {batch_count} 个批次")

    # 检查是否提取到了特征
    if len(features) == 0:
        print("错误: 没有提取到任何特征!")
        print("可能的原因:")
        print("1. 数据加载器为空")
        print("2. 所有批次都出现了错误")
        print("3. 模型没有正确输出")
        return np.array([]), np.array([]), np.array([])

    # 连接所有特征
    try:
        all_features = np.concatenate(features, axis=0)
        all_labels = np.concatenate(labels, axis=0)
        print(f"✅ 所有标签的唯一值: {np.unique(all_labels)}")  # ✅ 全局视角，这才是你真正要看的！
        all_sound_types = np.concatenate(sound_types, axis=0) if sound_types else np.zeros(len(all_labels), dtype=int)
        print(f"特征提取成功: 特征形状 {all_features.shape}, 标签形状 {all_labels.shape}")
        if sound_types:
            print(f"声音类型形状: {all_sound_types.shape}")
            print(f"唯一值: {np.unique(all_sound_types)}")
        return all_features, all_labels, all_sound_types

    except Exception as e:  # ✅ 捕获异常，并命名为 e
        print(f"连接特征时出错: {e}")  # ✅ 现在 e 是有效的
        return np.array([]), np.array([]), np.array([])


# 安全的UMAP绘图函数 - 修改为只显示calls和songs
def safe_plot_umap_main_task(features, main_labels, class_names, name, title='Main Task (Bird Species) UMAP Visualization'):
    """
    基于主任务（鸟类分类）进行 UMAP 可视化
    features: 模型输出的特征，numpy 数组，shape = [N, D]
    main_labels: 主任务类别标签，numpy 数组，shape = [N]，值为 0, 1, 2, ..., 类别总数-1
    class_names: 类别名称列表，比如 ['AMCR', 'AMRO', 'BLJA', ...]，长度等于类别数
    name: 保存的图片文件名
    title: 图片标题
    """
    if features.size == 0 or len(main_labels) == 0:
        print("错误: 特征或主任务标签为空，无法进行UMAP可视化")
        return

    if len(features) != len(main_labels):
        print(f"错误: 特征数量({len(features)})和主任务标签数量({len(main_labels)})不匹配")
        return

    print(f"开始主任务 UMAP可视化...")
    print(f"特征形状: {features.shape}, 主任务标签形状: {main_labels.shape}")
    print(f"主任务标签唯一值: {np.unique(main_labels)}")

    try:
        # 使用UMAP进行降维
        umap_model = umap.UMAP(n_components=2,
                           n_neighbors=30,  # 增加邻居数，使局部结构更加平滑
                           min_dist=0.9,  # 增大min_dist，增加类之间的间隔
                           random_state=42)
        umap_features = umap_model.fit_transform(features)

        # 类别名称，比如 ['AMCR', 'AMRO', 'BLJA', ...]，根据 bird_classes
        print(f"类别名称: {class_names}")
        unique_main_labels = np.unique(main_labels)
        print(f"主任务唯一标签: {unique_main_labels}")

        # 定义颜色映射 - 每个类别一个颜色
        color_mapping = {
            0: '#1f77b4',  # 蓝色
            1: '#ff7f0e',  # 橙色
            2: '#2ca02c',  # 绿色
            3: '#d62728',  # 红色
            4: '#9467bd',  # 紫色
            5: '#8c564b',  # 棕色
            6: '#e377c2',  # 粉色
            7: '#7f7f7f',  # 灰色
            8: '#bcbd22',  # 橄榄色
            9: '#17becf',  # 青色
        }

        # 绘制UMAP图
        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(main_labels)

        for label in unique_labels:
            if label not in color_mapping:
                color = '#000000'  # 默认黑色
            else:
                color = color_mapping[label]

            class_mask = (main_labels == label)
            class_count = np.sum(class_mask)
            if class_count > 0:  # 只绘制有样本的类别
                plt.scatter(umap_features[class_mask, 0],
                            umap_features[class_mask, 1],
                            color=color,
                            label=f"{class_names[label] if label < len(class_names) else 'Class_' + str(label)}",
                            s=20,
                            alpha=0.7)
        # plt.legend(
        #
        #     loc='upper right',  # 将图例放在左上角
        #     fontsize=26,  # 调整图例字体大小
        #     title_fontsize=20,  # 调整图例标题字体大小
        #     bbox_to_anchor=(1, 1),  # 设置图例相对于坐标轴的位置
        #     framealpha=0.3,
        #     prop={'family': 'Times New Roman', 'size': 26}  # 设置图例的字体
        # )
        # 自定义刻度仅显示 0 和 10 的倍数
        x_ticks = np.arange(0, np.ceil(umap_features[:, 0].max() / 10) * 10 + 1, 10)
        y_ticks = np.arange(0, np.ceil(umap_features[:, 1].max() / 10) * 10 + 1, 10)
        plt.xticks(ticks=x_ticks, fontsize=34, fontname='Times New Roman')  # 调整 x 轴刻度字体
        plt.yticks(ticks=y_ticks, fontsize=34, fontname='Times New Roman')  # 调整 y 轴刻度字体
        plt.tight_layout()
        plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
        print(f"主任务UMAP图已保存为: {name}")
        plt.show()

        # 打印统计信息
        print(f"\n📊 主任务类别统计:")
        for label in unique_labels:
            count = np.sum(main_labels == label)
            percentage = (count / len(main_labels)) * 100
            class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    except Exception as e:
        print(f"UMAP可视化时出错: {e}")


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载已训练好的模型
    try:

        model = MultiBranchCNN_G_M_WI(num_classes=10).to(device)
        model.load_state_dict(torch.load("Resnet_GM_OverS_Mixstyle123_mutitask_d3_best_avg_vaL_111.pth", map_location=device))
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # 检查region_paths和bird_classes是否正确定义
    print(f"region_paths: {region_paths}")
    print(f"bird_classes: {bird_classes}")

    # 修正数据集加载 - 根据BirdDataset的实际参数
    try:
        # 根据BirdDataset的定义，可能需要不同的参数
        test_dataset_region3 = BirdDataset(region_path=region_paths['region2'], bird_classes=bird_classes)
        test_loader_region3 = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)
        print(f"测试数据集加载成功，样本数: {len(test_dataset_region3)}")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("尝试使用备用参数...")
        try:
            # 备用参数尝试
            test_dataset_region3 = BirdDataset('region2', region_paths['region2'], {})
            test_loader_region3 = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)
            print(f"备用参数加载成功，样本数: {len(test_dataset_region3)}")
        except Exception as e2:
            print(f"备用参数也失败: {e2}")
            exit()

    # 提取特征 - 现在会尝试获取sound_types
    print("\n" + "=" * 50)
    print("开始特征提取")
    print("=" * 50)

    features_region3, labels_region3, sound_types_region3 = extract_features(model, test_loader_region3, device)

    # 数据加载器获取sound_types，尝试从文件名推断
    if sound_types_region3.size == 0 or np.all(sound_types_region3 == 0):
        print("\n尝试从文件名推断声音类型...")
        try:
            # 重新加载数据集以获取文件名
            test_dataset_for_filenames = BirdDataset(region_path=region_paths['region2'], bird_classes=bird_classes)
            all_filenames = []
            for i in range(len(test_dataset_for_filenames)):
                sample = test_dataset_for_filenames[i]
                if len(sample) >= 2:
                    if len(sample) == 5:
                        (_, _), _, filename, _, _ = sample
                    elif len(sample) == 4:
                        (_, _), _, filename, _ = sample
                    else:
                        (_, _), _ = sample
                        filename = f"sample_{i}"
                    all_filenames.append(filename)

            else:
                print("无法获取文件名，使用默认分类")
                sound_types_region3 = np.zeros(len(features_region3), dtype=int)
        except Exception as e:
            print(f"从文件名推断失败: {e}")
            sound_types_region3 = np.zeros(len(features_region3), dtype=int)

    # 检查特征提取结果
    if features_region3.size == 0 or labels_region3.size == 0:
        print("错误: 特征提取失败，无法继续可视化")
        exit()

    print(f"\n提取的特征信息:")
    print(f"特征形状: {features_region3.shape}")
    print(f"标签形状: {labels_region3.shape}")

    # UMAP可视化 - 只显示calls和songs
    print("\n" + "=" * 50)

    print("=" * 50)

    safe_plot_umap_main_task(features_region3,  labels_region3, bird_classes,
                               r"HARL-MOS-D3D2.png", r"HARL-MOS-$D_3S_2$")
