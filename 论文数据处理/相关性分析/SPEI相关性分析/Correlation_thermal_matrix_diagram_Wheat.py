import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与预处理 ---
# 确保路径指向小麦的 SPEI 相关性文件
data_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 修改点：聚焦冬/春小麦的关键水分敏感期 (当年时段) ---
# P14 (4月中旬) 到 P25 (7月中旬) 完整覆盖了冬小麦灌浆和春小麦抽穗扬花期
target_periods = list(range(14, 26))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 2. 构建数据矩阵 ---
matrix_r = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 按省份平均相关性排序（降序）
# R值越高，说明降水/水分对产量的正向促进越显著，即该省份对水分越敏感
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
            # 显著性标注：P < 0.01 (**), P < 0.05 (*)
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

# 12个周期，宽度设为 12-14 即可，视觉比例更佳
plt.figure(figsize=(14, 10), dpi=300)

# 使用 YlOrRd 色带，颜色越深代表水分限制越严重
sns.heatmap(matrix_r,
            cmap='YlOrRd',
            annot=annot_text.values,
            fmt="",
            linewidths=0.5,
            cbar_kws={'label': '相关系数 (R)'},
            vmin=0, vmax=0.6)

# --- 5. 自定义横轴标签 (增加小麦类型提示) ---
def get_wheat_stage_label(p):
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    # 标注小麦类型以增加学术解释力
    type_tag = ""
    if 14 <= p <= 20: type_tag = "\n(冬小麦关键期)"
    if 21 <= p <= 25: type_tag = "\n(春小麦关键期)"
    return f"{int(month)}月-{int(period_in_month)}期{type_tag}"

plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_wheat_stage_label(c) for c in matrix_r.columns],
           rotation=0, ha='center', fontsize=9)

# --- 6. 完善与保存 ---
plt.title('小麦产量与SPEI相关系数热力图 (4月中-7月中)', fontsize=20, pad=25)
plt.xlabel('生育周期 (8-day Periods)', fontsize=14)
plt.ylabel('省份 (按水分敏感度排序)', fontsize=14)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Wheat_SPEI_Heatmap_Critical_Periods.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 小麦热力图已聚焦关键期：P14-P25，保存在: {save_path}")