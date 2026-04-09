import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与预处理 ---
data_path = r'E:\data\计算结果\相关性分析_SVPD\Rice_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps_SVPD'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 2. 聚焦水稻关键敏感期 (早/中/晚稻抽穗灌浆期) ---
# P22(6月中旬) 到 P37(10月中旬) 完整覆盖了不同类型水稻的关键发育阶段
target_periods = list(range(22, 38))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 3. 构建数据矩阵并修正排序逻辑 ---
matrix_r = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 【重要修正】：计算平均相关系数，并按“升序”排列 (ascending=True)
# SVPD 的负相关越强 (值越小，如 -0.6)，代表高温干热胁迫越严重，应排在最上方
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
            # 标注显著性：P < 0.01 (**), P < 0.05 (*)
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

plt.figure(figsize=(16, 10), dpi=300)

# 【重要修正】：使用 Reds_r 色带，并将 vmin 设为负值
# 越红代表负相关越强（热害/干热胁迫越重），0 附近为白色
sns.heatmap(matrix_r,
            cmap='Reds_r',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)\n(负值代表高温干热减产倾向)'},
            vmin=-0.6, vmax=0)

# --- 6. 自定义横轴标签 ---
def get_rice_stage_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    type_tag = ""
    if 22 <= p <= 25: type_tag = "\n(早稻关键期)"
    elif 27 <= p <= 31: type_tag = "\n(中稻关键期)"
    elif 33 <= p <= 37: type_tag = "\n(晚稻关键期)"
    return f"{int(month)}月-{int(period_in_month)}期{type_tag}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_rice_stage_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 7. 完善与保存 ---
plt.title('水稻产量与 SVPD (饱和水汽压差) 相关系数热力图(6月中-10月中)', fontsize=20, pad=25)
plt.xlabel('生育周期 (8-day Periods)', fontsize=14)
plt.ylabel('省份 (按干热敏感度排序：越靠上越敏感)', fontsize=14)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Rice_SVPD_Heatmap_Critical_Final.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 水稻 SVPD 热力图已修正并聚焦关键期：P22-P37")