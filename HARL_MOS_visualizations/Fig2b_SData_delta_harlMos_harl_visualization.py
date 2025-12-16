import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 Excel 文件
file_path = "S1S2CompHARL-MOS-with-HARLF1.xlsx"
excel_file = pd.ExcelFile(file_path)
df_raw = excel_file.parse(sheet_name=0, skiprows=2)

# 2. 提取 HARL-MOS 数据 (保持原代码提取逻辑)
df_harlmos = df_raw.iloc[1:3, 2:].copy()
df_harlmos.columns = df_raw.iloc[0, 2:].tolist()
df_harlmos.insert(0, "Domain", df_raw.iloc[1:3, 1].tolist())
df_harlmos = df_harlmos.dropna(axis=1, how="all")
df_harlmos.iloc[:, 1:] = df_harlmos.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 3. 提取 HARL 数据 (保持原代码提取逻辑)
df_harl = df_raw.iloc[5:7, 2:].copy()
df_harl.columns = df_raw.iloc[4, 2:].tolist()
df_harl.insert(0, "Domain", df_raw.iloc[5:7, 1].tolist())
df_harl = df_harl.dropna(axis=1, how="all")
df_harl.iloc[:, 1:] = df_harl.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 4. 计算 Delta（HARL-MOS - HARL）
delta_df = df_harlmos.set_index("Domain") - df_harl.set_index("Domain")
delta_df = delta_df.dropna(axis=1, how="any")

# 5. 准备变量
domains = delta_df.index.tolist()
metrics = delta_df.columns.tolist()
x = np.arange(len(metrics))
width = 0.14

# 6. 计算每个指标的平均 Delta (移除平滑曲线，改为参照代码的离散点逻辑)
average_deltas = delta_df.mean(axis=0)          # Series
values = average_deltas.values
x_pos_for_mean = x + width * (len(domains) - 1) / 2   # 均值标记放在每组柱子的正中间

# 7. 开始绘图
fig, ax = plt.subplots(figsize=(14, 10))

# 分组柱状图
for i, domain in enumerate(domains):
    ax.bar(x + i * width, delta_df.loc[domain].values, width=width, label=domain)

# 均值：空心圆圈 + 数值标注 (严格参照提供的格式)
ax.scatter(x_pos_for_mean, values,
           s=200,                    # 点的大小
           facecolors='none',        # 空心
           edgecolors='black',
           linewidth=3,
           zorder=6,
           label="Mean Δ")           # 图例里会显示这个

# 均值数值标注（稍微向上偏移，避免和点重叠）
for xi, yi in zip(x_pos_for_mean, values):
    ax.text(xi, yi + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # 自适应偏移
            f"{yi:.2f}",
            ha='center', va='bottom', fontsize=26, fontweight='bold')

# 设置坐标轴
ax.set_xticks(x + width * (len(domains) - 1) / 2)
ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# 加粗坐标轴
for spine in ax.spines.values():
    spine.set_linewidth(3)

# 标签、标题、零线
ax.set_ylabel("HARL-MOS − HARL", fontsize=28, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=2.5)

# 图例（把 Mean Δ 也放进去）
ax.legend(title="Domain / Mean",
          loc='upper right',
          fontsize=24,
          title_fontsize=26,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout()
plt.show()