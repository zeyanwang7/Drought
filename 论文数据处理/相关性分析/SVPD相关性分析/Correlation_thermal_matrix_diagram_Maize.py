import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与预处理 ---
data_path = r'E:\data\计算结果\相关性分析_SVPD\Maize_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps_SVPD'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 2. 聚焦春夏玉米共同的关键水分敏感期 (抽雄-灌浆期) ---
# P23(6月下) - P31(8月下) 是中国玉米受高温热害和大气干旱最剧烈的窗口
target_periods = list(range(23, 32))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 3. 构建数据矩阵并修正排序逻辑 ---
matrix_r = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 【重要修正】：按省份平均相关性排序
# SVPD 与产量呈负相关。R 值越小（即负值越大，如 -0.55）代表干热胁迫越严重。
# 使用 ascending=True 确保受灾最严重的省份排在矩阵最上方。
matrix_r['avg'] = matrix_r.mean(axis=1)
matrix_r = matrix_r.sort_values(by='avg', ascending=True).drop(columns='avg')
matrix_p = matrix_p.reindex(matrix_r.index)

# --- 4. 生成带星号的标注文本 ---
annot_text = matrix_r.copy().astype(str)
for i in range(matrix_r.shape[0]):
    for j in range(matrix_r.shape[1]):
        val_r = matrix_r.iloc[i, j]
        val_p = matrix_p.iloc[i, j]
        if pd.isna(val_r):
            annot_text.iloc[i, j] = ""
        else:
            # 显著性水平标注
            if val_p < 0.01:
                star = "**"
            elif val_p < 0.05:
                star = "*"
            else:
                star = ""
            annot_text.iloc[i, j] = f"{val_r:.2f}{star}"

# --- 5. 绘图配置 (针对 SVPD 的负向效应优化) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 10), dpi=300)

# 【重要修正】：
# 1. 使用 Reds_r 色带（红色反向），使显著的负相关（干热灾害）显示为深红色。
# 2. 设置 vmin 为负值（如 -0.6），vmax 为 0，精准捕捉减产信号。
sns.heatmap(matrix_r,
            cmap='Reds_r',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)\n(负值代表大气干热导致的减产倾向)'},
            vmin=-0.6, vmax=0)

# --- 6. 自定义横轴标签 ---
def get_month_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    stage = ""
    if 23 <= p <= 27: stage = "\n(春玉米关键期)"
    if 28 <= p <= 31: stage = "\n(夏玉米关键期)"
    return f"{int(month)}月-{int(period_in_month)}期{stage}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_month_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 7. 完善与保存 ---
plt.title('玉米产量与 SVPD 相关性热力图 (6月下-8月下)', fontsize=18, pad=20)
plt.xlabel('生育周期 (8-day Periods)', fontsize=12)
plt.ylabel('省份 (按干热敏感度升序排序：越靠上越敏感)', fontsize=12)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Maize_SVPD_Heatmap_Critical_Final.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 玉米 SVPD 热力图修正完成！聚焦期：P23-P31")