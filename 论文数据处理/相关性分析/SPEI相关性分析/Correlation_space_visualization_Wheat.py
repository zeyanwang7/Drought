import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
data_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
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

# --- 4. 构建时间映射与时段 ---
def get_month_period(p):
    """根据阶段ID返回具体的月份和期数"""
    month = (p - 1) // 4 + 1
    period_in_month = (p - 1) % 4 + 1
    return f"{int(month)}月-{int(period_in_month)}期"

# 设置目标时段：
# 10月中(38) 到 11月底(44) -> 约 7 个阶段
# 次年 3月(9) 到 5月底(20) -> 约 12 个阶段
target_periods = list(range(38, 45)) + list(range(9, 21))

# 总计 7 + 12 = 19 个阶段，建议使用 4行 5列 布局 (20个位置)
cols = 5
rows = 4
fig, axes = plt.subplots(rows, cols, figsize=(22, 18), dpi=300)
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

    # 设置标题（区分当年/次年）
    year_prefix = "上年" if p_id >= 37 else "当年"
    title_text = f"{year_prefix} {get_month_period(p_id)}"
    ax.set_title(title_text, fontsize=11, fontweight='bold')
    ax.axis('off')

# --- 6. 清理多余格子 ---
for j in range(len(target_periods), len(axes)):
    axes[j].axis('off')

# --- 7. 公共颜色条与保存 ---
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=0.6))
cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
fig.colorbar(sm, cax=cbar_ax, label='相关系数 (R)')

plt.suptitle('小麦产量对SPEI的敏感性空间分布 (播种期 & 返青-抽穗期)', fontsize=26, y=0.98)

# 保存文件
save_path = os.path.join(output_dir, 'Wheat_Map_Winter_Spring.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘制完成！已选取 {len(target_periods)} 个跨年阶段。")