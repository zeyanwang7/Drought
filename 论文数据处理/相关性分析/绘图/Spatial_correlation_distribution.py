import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 路径设置 ---
spei_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
# 建议确认 SVPD 文件名是否正确，此处按照常见的命名逻辑设置
svpd_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
shapefile_path = r'E:\数据集\中国_省\中国_省2.shp'
output_dir = r'E:\data\计算结果\Maps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 预处理函数：计算关键期平均相关系数 ---
def get_avg_correlation(file_path, target_periods):
    if not os.path.exists(file_path):
        print(f"警告：找不到文件 {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['match_key'] = df['Province'].astype(str).str.strip().str[:2]
    # 筛选关键期：播种期(38-44) & 返青-抽穗期(9-20)
    df_target = df[df['Period'].isin(target_periods)].copy()
    # 计算省份平均值
    avg_df = df_target.groupby('match_key')['R'].mean().reset_index()
    return avg_df

# 定义目标关键期
target_periods =  list(range(9, 21))

# 计算两组数据的平均值
df_spei_avg = get_avg_correlation(spei_path, target_periods)
df_svpd_avg = get_avg_correlation(svpd_path, target_periods)

# --- 3. 准备地图数据 ---
china_map = gpd.read_file(shapefile_path, encoding='UTF-8')
china_map['match_key'] = china_map['name'].astype(str).str.strip().str[:2]

# 合并数据
map_spei = china_map.merge(df_spei_avg, on='match_key', how='left')
map_svpd = china_map.merge(df_svpd_avg, on='match_key', how='left')

# --- 4. 绘图配置 (1行2列) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)

# 定义统一的绘图参数
plot_params = {
    'cmap': 'YlOrRd',
    'vmin': 0,
    'vmax': 0.5,  # 统一刻度以便对比
    'edgecolor': 'white',
    'linewidth': 0.3
}

# 子图1：SPEI (水分敏感性)
china_map.plot(ax=ax1, color='#f5f5f5', edgecolor='darkgray', linewidth=0.5)
map_spei.plot(column='R', ax=ax1, **plot_params)
ax1.set_title('(a) SPEI (水分敏感性) 平均相关系数', fontsize=18, fontweight='bold', pad=15)
ax1.axis('off')

# 子图2：SVPD (干热敏感性)
china_map.plot(ax=ax2, color='#f5f5f5', edgecolor='darkgray', linewidth=0.5)
map_svpd.plot(column='R', ax=ax2, **plot_params)
ax2.set_title('(b) SVPD (干热敏感性) 平均相关系数', fontsize=18, fontweight='bold', pad=15)
ax2.axis('off')

# --- 5. 公共颜色条 ---
# 创建一个单独的轴存放颜色条
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=0.5))
cbar_ax = fig.add_axes([0.45, 0.08, 0.12, 0.025]) # [left, bottom, width, height]
fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='平均相关系数 (R)')

# --- 6. 完善布局并保存 ---
plt.suptitle('小麦关键生育阶段对水分(SPEI)与干热(SVPD)响应的空间差异对比', fontsize=26, y=1.02, fontweight='bold')
plt.figtext(0.5, 0.02, f'关键期范围：返青-抽穗期 (P9-20)',
            ha='center', fontsize=12, color='gray')

# 调整子图间距
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

save_path = os.path.join(output_dir, 'Wheat_SPEI_SVPD_Comparison_Map.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 对比地图已绘制完成！保存至: {save_path}")