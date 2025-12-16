import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import tensor
from torch.utils.data import DataLoader
from statsmodels.stats.contingency_tables import mcnemar

# -------------------------------
# 1. McNemar 检验函数
# -------------------------------
def mcnemar_test(y_true_indices, y_pred1, y_pred2):
    """
    执行 McNemar 检验
    参数:
        y_true_indices: 真实标签的整数索引 (如 [0, 1, 2, ...])
        y_pred1: 模型1预测的整数索引 (如 [0, 1, 2, ...])
        y_pred2: 模型2预测的整数索引 (如 [0, 1, 2, ...])
    """
    # ✅ 确保所有输入都是整数数组
    # y_true_indices = np.array(y_true_indices, dtype=np.int64)
    # y_pred1 = np.array(y_pred1, dtype=np.int64)
    # y_pred2 = np.array(y_pred2, dtype=np.int64)


    correct1 = (y_pred1 == y_true_indices)
    correct2 = (y_pred2 == y_true_indices)

    b = np.sum(correct2 & ~correct1)  # model2 对，model1 错
    c = np.sum(~correct2 & correct1)  # model1 对，model2 错

    table = np.array([[0, b], [c, 0]])

    result = mcnemar(table, exact=True)
    p_value = result.pvalue



    return p_value, b, c

# -------------------------------
# 2. 模型推理函数
# -------------------------------
def get_model_predictions(model, test_loader, device="cuda"):
    model.eval()
    model.to(device)
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs_list, labels = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # ✅ 转换为 float32
            if isinstance(inputs_list, list):
                inputs_list = [inp.to(device).float() for inp in inputs_list]
            elif isinstance(inputs_list, torch.Tensor):
                inputs_list = inputs_list.to(device).float()
            else:
                raise TypeError(f"inputs_list must be list or Tensor, got {type(inputs_list)}")

            # ✅ 解包并传入模型
            if len(inputs_list) == 2:
                mel, gamma = inputs_list
                outputs = model(mel, gamma)  # ❗ outputs 可能是 tuple
            else:
                raise ValueError(f"Expected 2 inputs (mel, gamma), but got {len(inputs_list)}")



            # ✅ 关键修复：如果 outputs 是 tuple，取第一个元素（通常是 logits）
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # ✅ 取主输出（分类 logits）

            # ✅ 现在 outputs 应该是 Tensor
            if not isinstance(outputs, torch.Tensor):
                raise TypeError(f"Model output should be Tensor or tuple, got {type(outputs)}")

            _, preds = torch.max(outputs, dim=1)  # ✅ 现在可以正常工作
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)
# -------------------------------
# 3. 加载模型（请根据实际类名修改！）
# -------------------------------
def load_your_model(model_path, num_classes, device, model_type="harl"):
    from MG_OverSampling_ResNet_Multitask import MultiBranchCNN_G_M_WI as HARL_OS
    from MG_OverSampling_MixSty_ResNet_Multitask_D import MultiBranchCNN_G_M_WI as HARL_MOS

    if model_type.lower() in ["harl_mos", "mos"]:
        model = HARL_MOS(num_classes=num_classes)
    elif model_type.lower() == "harl":
        model = HARL_OS(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}，请选择 'harl' 或 'harl_mos'")

    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_true_labels_with_consistency_check(gamma_path, mel_path, bird_classes):
    """
    从 gamma 和 mel 文件夹提取真实标签，并检查两者结构一致性
    """
    true_labels = []

    # 获取 gamma 和 mel 的子文件夹（排序以保证顺序一致）
    gamma_subfolders = sorted([
        d for d in os.listdir(gamma_path) if os.path.isdir(os.path.join(gamma_path, d))
    ])
    mel_subfolders = sorted([
        d for d in os.listdir(mel_path) if os.path.isdir(os.path.join(mel_path, d))
    ])

    # ✅ 检查子文件夹是否一致
    if gamma_subfolders != mel_subfolders:
        print("⚠️ 警告：gamma 和 mel 的子文件夹不一致！")
        print(f"Gamma: {gamma_subfolders}")
        print(f"Mel:   {mel_subfolders}")
        # 取交集，避免崩溃
        common_classes = set(gamma_subfolders) & set(mel_subfolders)
        gamma_subfolders = sorted(common_classes)
        mel_subfolders = sorted(common_classes)
        print(f"✅ 仅使用共同类别: {gamma_subfolders}")

    # 构建类别映射
    bird_class_map = {name: idx for idx, name in enumerate(bird_classes)}

    # 遍历每个类别
    for class_dir in gamma_subfolders:
        gamma_class_path = os.path.join(gamma_path, class_dir)
        mel_class_path = os.path.join(mel_path, class_dir)

        # 获取 .npy 文件（假设特征文件为 .npy）
        gamma_files = sorted([f for f in os.listdir(gamma_class_path) if f.endswith('.mat')])
        mel_files = sorted([f for f in os.listdir(mel_class_path) if f.endswith('.mat')])

        # ✅ 检查文件数量是否一致
        if len(gamma_files) != len(mel_files):
            print(f"⚠️ 类别 {class_dir}: gamma 有 {len(gamma_files)} 个文件，mel 有 {len(mel_files)} 个文件，数量不匹配！")
            # 取最小数量，避免越界
            min_count = min(len(gamma_files), len(mel_files))
            gamma_files = gamma_files[:min_count]
            mel_files = mel_files[:min_count]
        elif len(gamma_files) == 0:
            print(f"⚠️ 类别 {class_dir} 中没有 .npy 文件，跳过")
            continue

        # ✅ 清理类别名（下划线转空格）
        clean_name = class_dir.replace('_', ' ')
        # clean_name = ' '.join(word.capitalize() for word in clean_name.split())  # 可选大写

        # ✅ 检查是否在 bird_classes 中
        if clean_name not in bird_class_map:
            print(f"⚠️ 警告：{clean_name} 不在 bird_classes 中，跳过")
            continue

        # ✅ 添加真实标签（每个样本一个标签）
        true_labels.extend([clean_name] * len(gamma_files))

    return true_labels
# -------------------------------
# 4. 主程序
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================
    # ✅ 你需要按需修改的部分：
    # ============================

    # --- 1. 导入你的 test_loader（必须返回 (inputs, labels)）
    from MG_OverSampling_MixSty_ResNet_Multitask_D import load_data   # ✅ 请确认是否正确！

    # --- 2. 鸟类类别和类别数
    bird_classes = [
        "Agelaius phoeniceus", "Cardinalis cardinalis", "Certhia americana",
        "Corvus brachyrhynchos", "Molothrus ater", "Setophaga aestiva",
        "Setophaga ruticilla", "Spinus tristis", "Tringa semipalmata", "Turdus migratorius"
    ]
    num_classes = len(bird_classes)
    # 定义路径和类别
    gamma_path = r"F:\NF2\gamma\data_wav_8s_2"
    mel_path = r"F:\NF2\mel\data_wav_8s_2"

    region_paths = {
        'region1': {'gamma': os.path.join(gamma_path, '1'), 'mel': os.path.join(mel_path, '1')},
        'region2': {'gamma': os.path.join(gamma_path, '2'), 'mel': os.path.join(mel_path, '2')},
        'region3': {'gamma': os.path.join(gamma_path, '3'), 'mel': os.path.join(mel_path, '3')}
    }

    gamma_region3_path = region_paths['region2']['gamma']  # F:\NF2\gamma\data_wav_8s_2\2
    mel_region3_path = region_paths['region2']['mel']  # F:\NF2\mel\data_wav_8s_2\2

    gamma_region1_path = region_paths['region1']['gamma']  # F:\NF2\gamma\data_wav_8s_2\2
    mel_region1_path = region_paths['region1']['mel']  # F:\NF2\mel\data_wav_8s_2\2

    # ✅ 使用双路径提取真实标签，并检查一致性
    y_true_labels = get_true_labels_with_consistency_check(
        gamma_path=gamma_region1_path,
        mel_path=mel_region1_path,
        bird_classes=bird_classes
    )
    y_true_labels1 = get_true_labels_with_consistency_check(
        gamma_path=gamma_region3_path,
        mel_path=mel_region3_path,
        bird_classes=bird_classes
    )
    print(f"✅ 成功提取 {len(y_true_labels)} 个真实标签")
    print(f"前5个标签: {y_true_labels[:5]}")

    # ✅ 正确：调用 load_data 函数，传入 region_paths 和 bird_classes
    data_loaders = load_data(region_paths, bird_classes)
    # --- 3. 两个模型权重路径
    model1_path = "Resnet_GM_OverS_d3_best_avg_val.pth"
    model2_path = "Resnet_GM_OverS_Mixstyle_d3_best_avg_vaL_renew_lr_1e_8h_0.4.pth"

    # --- 4. 加载模型

    model1 = load_your_model(model1_path, num_classes, device, model_type="harl")
    model2 = load_your_model(model2_path, num_classes, device, model_type="harl_mos")

    # --- 5. 推理，得到预测
    y_pred1 = get_model_predictions(model1, data_loaders['test_region1'], device=device)
    y_pred2 = get_model_predictions(model2, data_loaders['test_region1'], device=device)

    y_pred3 = get_model_predictions(model1, data_loaders['test_region2'], device=device)
    y_pred4 = get_model_predictions(model2, data_loaders['test_region2'], device=device)




    # --- 5. 创建输出目录 ---
    output_dir = "HARL"
    os.makedirs(output_dir, exist_ok=True)



    # y_true_labels = get_true_labels_with_consistency_check(gamma_region2_path, mel_region2_path, bird_classes)

    # 确保长度一致
    assert len(y_true_labels) == len(y_pred1) == len(y_pred2), "样本数不一致！"
    assert len(y_true_labels1) == len(y_pred3) == len(y_pred4), "样本数不一致！"



    # --- 7. 执行 McNemar 检验
    label_to_idx = {label: idx for idx, label in enumerate(bird_classes)}
    y = [label_to_idx[label] for label in y_true_labels]
    y1 = [label_to_idx[label] for label in y_true_labels1]

    # ✅ 确保长度一致
    # assert len(y_true) == len(y_pred1) == len(y_pred2), \
    #     f"样本数不一致！y_true: {len(y_true)}, y_pred1: {len(y_pred1)}, y_pred2: {len(y_pred2)}"

    # ✅ 执行 McNemar 检验（用整数索引比较）
    p_value, b, c = mcnemar_test(y, y_pred1, y_pred2)
    print(f"🔍 McNemar 检验 p-value: {p_value:.4f}")
    print(f"✅ 模型HARL-MOS正确但模型HARL错误: {b}")
    print(f"❌ 模型HARL-MOS错误但模型HARL正确: {c}")

    if p_value < 0.05:
        print("🎯 结论：两个模型预测结果存在显著差异（p < 0.05）")
    else:
        print("⚠️ 结论：两个模型预测结果差异不显著（p >= 0.05）")



    df_comparison = pd.DataFrame({
        "true_label": y,
        "model1_pred_index": y_pred1,
        # "model1_pred_label": y_pred1_labels,
        "model2_pred_index": y_pred2,
        # "model2_pred_label": y_pred2_labels
    })
    df_comparison.to_csv(
        os.path.join(output_dir, "HARL-MO＆HARL-MOS-D3D1-D3D1-predictions_comparison.csv"),
        index=False
    )

    df_comparison1 = pd.DataFrame({
        "true_label": y1,
        "model1_pred_index": y_pred3,
        # "model1_pred_label": y_pred1_labels,
        "model2_pred_index": y_pred4,
        # "model2_pred_label": y_pred2_labels
    })
    df_comparison1.to_csv(
        os.path.join(output_dir, "HARL-MO＆HARL-MOS-D3D2-D3D2-predictions_comparison.csv"),
        index=False
    )