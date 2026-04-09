import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 数据加载与预处理 ---
# 使用 SVPD 相关性结果文件
data_path = r'E:\data\计算结果\相关性分析_SVPD\Cotton_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\Maps_SVPD'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_corr = pd.read_csv(data_path)

# --- 修改点 1: 定义棉花关键生育期 (4月上旬 - 10月下旬) ---
# 8-day 周期计算：4月1日约为第 13 期，10月底约为第 38 期
# 范围涵盖了 播种、苗期、蕾期、花铃期和吐絮期
target_periods = list(range(13, 39))
df_sub = df_corr[df_corr['Period'].isin(target_periods)].copy()

# --- 2. 构建数据矩阵并强制排序 ---
matrix_r_raw = df_sub.pivot(index='Province', columns='Period', values='R')
matrix_p_raw = df_sub.pivot(index='Province', columns='Period', values='P_Value')

# 按照时间轴重排各列
matrix_r = matrix_r_raw.reindex(columns=target_periods)
matrix_p = matrix_p_raw.reindex(columns=target_periods)

# 按省份平均相关性排序
# SVPD 与产量通常为负相关，均值越小（负值越大）代表对干热越敏感
matrix_r['avg'] = matrix_r.mean(axis=1)
matrix_r = matrix_r.sort_values(by='avg', ascending=True).drop(columns='avg')
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
            # 显著性 P < 0.05 标注星号，P < 0.01 标注双星（可选）
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

plt.figure(figsize=(18, 10), dpi=300)

# 使用 Reds_r 色带：负相关越强颜色越红
sns.heatmap(matrix_r,
            cmap='Reds_r',
            annot=annot_text.values,
            fmt="",
            linewidths=0.3,
            cbar_kws={'label': '相关系数 (R)\n(负值代表大气干热导致的减产趋势)'},
            vmin=-0.8, vmax=0)


# --- 修改点 2: 自定义横轴标签 (适应棉花单年内生育期) ---
def get_cotton_label(p):
    # 计算月份：每个月约为 4 个 8-day 周期
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    # 标注具体的生育阶段参考（可选）
    stage = ""
    if 13 <= p <= 16:
        stage = "(播种苗期)"
    elif 21 <= p <= 28:
        stage = "(花铃期)"
    elif p >= 35:
        stage = "(吐絮期)"

    return f"{int(month)}月-{int(period_in_month)}期{stage}"


plt.xticks(ticks=[i + 0.5 for i in range(len(matrix_r.columns))],
           labels=[get_cotton_label(c) for c in matrix_r.columns],
           rotation=45, ha='right', fontsize=9)

# --- 5. 完善与保存 ---
plt.title('棉花产量与 SVPD (饱和水汽压差) 相关系数热力矩阵\n(4月播种期 - 10月吐絮收获期)', fontsize=20, pad=20)
plt.xlabel('生育周期 (8-day Periods)', fontsize=14)
plt.ylabel('省份 (按干热敏感度排序)', fontsize=14)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Cotton_SVPD_Heatmap_Growing_Season.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 棉花 SVPD 热力图已生成！")
print(f"保存路径: {save_path}")