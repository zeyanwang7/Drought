import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与预处理 ---
data_path = r'E:\data\计算结果\相关性分析\Rice_SPEI_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 修改点：聚焦水稻关键水分敏感期 ---
# P22(6月中旬) 到 P37(10月中旬) 完整覆盖了早、中、晚三类水稻的抽穗-灌浆临界时段
target_periods = list(range(22, 38))
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
            # 标注显著性：P < 0.01 (**), P < 0.05 (*)
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

# 时间段由 29 个缩减为 16 个，宽度设为 14-16 即可，更清晰
plt.figure(figsize=(16, 10), dpi=300)

# 使用 YlOrRd 色带，R正值越大颜色越深（反映水分亏缺限制产量的程度）
sns.heatmap(matrix_r,
            cmap='YlOrRd',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)'},
            vmin=0, vmax=0.6)

# --- 5. 自定义横轴标签 (增加水稻类型提示) ---
def get_rice_stage_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    # 标注水稻类型以增加可读性
    type_tag = ""
    if 22 <= p <= 25: type_tag = "\n(早稻)"
    elif 27 <= p <= 31: type_tag = "\n(中稻)"
    elif 33 <= p <= 37: type_tag = "\n(晚稻)"
    return f"{int(month)}月-{int(period_in_month)}期{type_tag}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_rice_stage_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 6. 完善与保存 ---
plt.title('水稻产量与SPEI相关系数热力图 (6月中-10月中)', fontsize=20, pad=25)
plt.xlabel('生育周期 (8-day Periods)', fontsize=14)
plt.ylabel('省份 (按水分敏感度排序)', fontsize=14)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Rice_Heatmap_Critical_Periods.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 热力图已聚焦水稻关键期：P22-P37，保存在: {save_path}")