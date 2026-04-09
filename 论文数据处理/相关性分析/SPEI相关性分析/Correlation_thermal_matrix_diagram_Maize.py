import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与预处理 ---
data_path = r'E:\data\计算结果\相关性分析\Maize_SPEI_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 修改点：仅选择春夏玉米共同的关键水分敏感期 ---
# P23(6月下) - P31(8月下) 涵盖了春玉米和夏玉米的抽雄-灌浆临界期
target_periods = list(range(23, 32))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 2. 构建数据矩阵 ---
matrix_r = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 按省份平均相关性排序（降序）
matrix_r['avg'] = matrix_r.mean(axis=1)
matrix_r = matrix_r.sort_values(by='avg', ascending=False).drop(columns='avg')
matrix_p = matrix_p.reindex(matrix_r.index)

# --- 3. 生成带星号的标注文本 ---
annot_text = matrix_r.copy().astype(str)
for i in range(matrix_r.shape[0]):
    for j in range(matrix_r.shape[1]):
        val_r = matrix_r.iloc[i, j]
        val_p = matrix_p.iloc[i, j]
        if pd.isna(val_r):
            annot_text.iloc[i, j] = ""
        else:
            # 显著性水平标注：P < 0.01 (**), P < 0.05 (*)
            if val_p < 0.01:
                star = "**"
            elif val_p < 0.05:
                star = "*"
            else:
                star = ""
            annot_text.iloc[i, j] = f"{val_r:.2f}{star}"

# --- 4. 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 时间段缩短后，画布宽度可以适当缩小，设为 12x10 更加紧凑
plt.figure(figsize=(12, 10), dpi=300)

# 使用 YlOrRd 色带，R值（正相关）越大颜色越深，代表水分限制越严重
sns.heatmap(matrix_r,
            cmap='YlOrRd',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)'},
            vmin=0, vmax=0.6)

# --- 5. 自定义横轴标签 ---
def get_month_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    # 标注具体的生育阶段
    stage = ""
    if 23 <= p <= 27: stage = "\n(春玉米关键期)"
    if 27 <= p <= 31: stage = "\n(夏玉米关键期)"
    return f"{int(month)}月-{int(period_in_month)}期{stage}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_month_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 6. 完善与保存 ---
plt.title('玉米产量与SPEI相关性热力图(6月下-8月下)', fontsize=18, pad=20)
plt.xlabel('关键生育期 (8-day Periods)', fontsize=12)
plt.ylabel('省份 (按水分敏感度降序排列)', fontsize=12)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Maize_Heatmap_Critical_Periods.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 热力图已聚焦关键期：6月下-8月下 (P23-P31)")