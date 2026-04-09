import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 路径设置 ---
# 使用之前生成的 SVPD 相关性结果文件
data_path = r'E:\data\计算结果\相关性分析_SVPD\Maize_SVPD_Yield_Correlation.csv'
shapefile_path = r'E:\数据集\中国_省\中国_省2.shp'
output_dir = r'E:\data\计算结果\Maps_SVPD'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 加载数据 ---
print("正在加载地图与相关性数据...")
china_map = gpd.read_file(shapefile_path, encoding='UTF-8')
df_corr = pd.read_csv(data_path)

# --- 3. 预处理：统一匹配键 (取省名前两个字) ---
china_map['match_key'] = china_map['name'].astype(str).str.strip().str[:2]
df_corr['match_key'] = df_corr['Province'].astype(str).str.strip().str[:2]

# --- 4. 构建时间映射函数 ---
def get_month_period(p):
    """根据阶段ID(1-46)返回具体的月份和期数"""
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    return f"{int(month)}月-{int(period_in_month)}期"

# 设定绘制范围：3月(9) 到 8月中旬(30) 以及 10月中旬(38) 到 11月底(44)
target_periods = list(range(9, 31)) + list(range(38, 45))

# 布局设置：5行 6列
cols = 6
rows = 5
fig, axes = plt.subplots(rows, cols, figsize=(25, 20), dpi=300)
axes = axes.flatten()

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 5. 循环绘图 ---
print("正在生成各阶段空间分布图...")
# 使用反转的红色系，因为我们关注的是负相关 (R < 0 表示 VPD 越高产量越低)
# 或者使用 'coolwarm' 并在 0 处对称
cmap_name = 'Reds_r'

for i, p_id in enumerate(target_periods):
    ax = axes[i]
    # 筛选当前阶段数据
    current_data = df_corr[df_corr['Period'] == p_id].copy()
    merged = china_map.merge(current_data, on='match_key', how='left')

    # 绘制灰色底图（代表无数据或非研究区）
    china_map.plot(ax=ax, color='#f5f5f5', edgecolor='darkgray', linewidth=0.3)

    # 绘制相关性空间分布
    # 我们将 vmin 设为 -0.6，vmax 设为 0，专门突出负相关显著的区域
    if not merged['R'].isna().all():
        merged.plot(column='R', ax=ax, cmap=cmap_name, vmin=-0.6, vmax=0,
                    edgecolor='white', linewidth=0.2)

    # 设置标题
    ax.set_title(get_month_period(p_id), fontsize=12, fontweight='bold')
    ax.axis('off')

# --- 6. 清理多余格子 ---
for j in range(len(target_periods), len(axes)):
    axes[j].axis('off')

# --- 7. 公共颜色条与保存 ---
# 颜色条范围从 -0.6 到 0
sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=-0.6, vmax=0))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
fig.colorbar(sm, cax=cbar_ax, label='相关系数 (R) [负值表示干热减产趋势]')

plt.suptitle('玉米产量对 SVPD (大气干热压力) 的敏感性空间分布 (3-8月 & 10-11月)', fontsize=28, y=0.98)

# 保存文件
save_path = os.path.join(output_dir, 'Maize_SVPD_Map_Monthly_Labels.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘图完成！图片保存在: {save_path}")