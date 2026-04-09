import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 路径与字体设置 ---
# 注意：确保文件是 SVPD 的相关性结果
data_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps_SVPD'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'MicroSoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df_corr = pd.read_csv(data_path)

# --- 2. 聚焦冬/春小麦的关键水分敏感期 (4月中-7月中) ---
# P14 (4月中旬) 到 P25 (7月中旬) 涵盖了北方冬小麦灌浆期和西北/东北春小麦抽穗期
target_periods = list(range(14, 26))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 3. 构建数据矩阵并修正排序逻辑 ---
matrix_r = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 【重要修正】：计算平均相关系数，并按“升序”排列
# 在 SVPD 中，R 值越小（负值越大，如 -0.6）代表干热胁迫对产量的负面影响越严重
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
            # 标注显著性水平：P < 0.01 (**), P < 0.05 (*)
            if val_p < 0.01:
                star = "**"
            elif val_p < 0.05:
                star = "*"
            else:
                star = ""
            annot_text.iloc[i, j] = f"{val_r:.2f}{star}"

# --- 5. 绘图配置 (针对 SVPD 的负向效应优化) ---
plt.figure(figsize=(14, 10), dpi=300)

# 【重要修正】：使用 Reds_r 色带
# Reds_r 会将绝对值大的负数标为深红色，0 附近标为白色
# vmin 设为 -0.6 (或数据中的最小值)，vmax 设为 0
sns.heatmap(matrix_r,
            cmap='Reds_r',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)\n(负值代表大气干热导致的减产倾向)'},
            vmin=-0.6, vmax=0)

# --- 6. 自定义横轴标签 (增加小麦类型提示) ---
def get_wheat_stage_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    type_tag = ""
    if 14 <= p <= 20: type_tag = "\n(冬小麦关键期)"
    if 21 <= p <= 25: type_tag = "\n(春小麦关键期)"
    return f"{int(month)}月-{int(period_in_month)}期{type_tag}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_wheat_stage_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 7. 完善与保存 ---
plt.title('小麦产量与 SVPD 相关系数热力矩阵(4月中-7月中)', fontsize=18, pad=25)
plt.xlabel('关键生育周期 (8-day Periods)', fontsize=14)
plt.ylabel('省份 (按干热敏感度排序：越靠上越敏感)', fontsize=14)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Wheat_SVPD_Heatmap_Critical_Final.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ SVPD 热力图修正完成！")
print(f"核心改进：排序逻辑已修正为升序（负值优先），色彩映射范围已修正为负值区间 [-0.6, 0]")