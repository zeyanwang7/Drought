import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
data_path = r'E:\data\计算结果\相关性分析\Rice_SPEI_Yield_Correlation.csv'
shapefile_path = r'E:\数据集\中国_省\中国_省2.shp'
output_dir = r'E:\data\计算结果\Maps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 加载数据 ---
china_map = gpd.read_file(shapefile_path, encoding='UTF-8')
df_corr = pd.read_csv(data_path)

# --- 3. 预处理：统一匹配键 ---
china_map['match_key'] = china_map['name'].astype(str).str.strip().str[:2]
df_corr['match_key'] = df_corr['Province'].astype(str).str.strip().str[:2]

# --- 4. 构建时间映射函数 ---
def get_month_period(p):
    """根据阶段ID返回具体的月份和期数"""
    # 逻辑：Period 1-4为1月, 5-8为2月, 9-12为3月...
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    return f"{int(month)}月-{int(period_in_month)}期"

# 段落1: 3月(9) 到 8月中旬(30)
# 段落2: 10月中旬(38) 到 11月底(44)
target_periods = list(range(9, 31)) + list(range(38, 45))

# 总计 29 个阶段，5行 6列 布局
cols = 6
rows = 5
fig, axes = plt.subplots(rows, cols, figsize=(25, 20), dpi=300)
axes = axes.flatten()

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 5. 循环绘图 ---
for i, p_id in enumerate(target_periods):
    ax = axes[i]
    current_data = df_corr[df_corr['Period'] == p_id].copy()
    merged = china_map.merge(current_data, on='match_key', how='left')

    # 绘制底图
    china_map.plot(ax=ax, color='#f5f5f5', edgecolor='darkgray', linewidth=0.3)

    # 绘制相关性
    if not merged['R'].isna().all():
        merged.plot(column='R', ax=ax, cmap='YlOrRd', vmin=0, vmax=0.6,
                    edgecolor='white', linewidth=0.2)

    # 设置具体的月份标题
    title_text = get_month_period(p_id)
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.axis('off')

# --- 6. 清理多余格子 ---
for j in range(len(target_periods), len(axes)):
    axes[j].axis('off')

# --- 7. 公共颜色条与保存 ---
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=0.6))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
fig.colorbar(sm, cax=cbar_ax, label='相关系数 (R)')

plt.suptitle('水稻产量对SPEI的敏感性空间分布 (3-8月 & 10-11月)', fontsize=28, y=0.98)

# 保存文件
save_path = os.path.join(output_dir, 'Rice_Map_Monthly_Labels.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘制完成！标题已更新为具体月份，保存在: {save_path}")