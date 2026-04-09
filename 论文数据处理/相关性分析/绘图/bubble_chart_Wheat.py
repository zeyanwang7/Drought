import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
# 切换为小麦的相关性计算结果
spei_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
svpd_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 数据加载与预处理 ---
def load_data(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Index'] = label
        return df
    else:
        print(f"找不到文件: {path}")
        return None

df_spei = load_data(spei_path, 'SPEI')
df_svpd = load_data(svpd_path, 'SVPD')

if df_spei is None or df_svpd is None: exit()

# 合并数据
df_all = pd.concat([df_spei, df_svpd], axis=0)

# 【核心调整】：筛选小麦关键期 (P12 - P28)
# P12-P24 主要是冬小麦拔节-灌浆；P18-P28 覆盖春小麦抽穗-灌浆
target_periods = list(range(12, 29))
df_plot = df_all[df_all['Period'].isin(target_periods)].copy()

# 处理 P 值：负对数转换。数值越大，点越大，代表越显著
df_plot['LogP'] = -np.log10(df_plot['P_Value'] + 1e-5)

# 限制 LogP 的显示范围，避免个别极显著点导致气泡大小失调
df_plot['Size_LogP'] = df_plot['LogP'].clip(0, 5)

# --- 3. 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用 relplot 绘制多省份气泡图
# hue="R"：颜色代表相关性强度（正负）
# size="Size_LogP"：气泡大小代表显著性
g = sns.relplot(
    data=df_plot,
    x="Period", y="Index",
    hue="R", size="Size_LogP",
    col="Province", col_wrap=4, # 每行显示4个省份
    palette="RdBu_r",           # 经典红蓝配色：红负相关，蓝正相关
    hue_norm=(-0.7, 0.7),       # 统一颜色刻度范围
    sizes=(20, 350),            # 气泡的最小和最大尺寸
    kind="scatter",
    height=3.2, aspect=1.3,
    edgecolor="0.3", linewidth=0.6,
    alpha=0.85                  # 稍微透明一点，重叠时更好看
)

# --- 4. 美化与细节调整 ---
g.set_titles("{col_name}", fontweight='bold', fontsize=13)
g.set_axis_labels("生育阶段 (8-day Period)", "气象指数强度")

# 优化横坐标刻度：小麦跨度大，每隔2个Period标一个数字
for ax in g.axes.flat:
    ax.set_xticks([12, 16, 20, 24, 28])
    # 添加一条辅助线分隔 SPEI 和 SVPD
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)

# 调整布局与标题
g.fig.subplots_adjust(top=0.92, hspace=0.35)
g.fig.suptitle('小麦产量与 SPEI / SVPD 时空相关性显著性图谱 (P12-P28)', fontsize=20, fontweight='bold')

# 修改图例标题
g._legend.set_title("显著性与相关性")
new_labels = ['R (Corr)', 'Sig (-log P)']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

# 保存文件
output_file = os.path.join(output_dir, 'Wheat_Bubble_Spatial_Correlation.png')
print("正在渲染高分辨率气泡图...")
g.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 任务完成！小麦气泡图已保存至: {output_file}")