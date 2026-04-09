import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import os
import warnings

# --- 0. 环境设置 ---
warnings.filterwarnings("ignore")

# --- 1. 路径配置 ---
syrs_path = r'E:\data\计算结果\表格\单产标准化残差_SYRS.xlsx'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
save_map_path = r'E:\data\计算结果\表格\SYRI_Final_Combined_Terra.png'

crops = ['小麦', '玉米', '水稻']
# 换回您喜欢的经典 Terra 高级色系
colors_list = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']


# --- 2. 字体加载 ---
def get_chinese_font():
    paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']
    for p in paths:
        if os.path.exists(p):
            return font_manager.FontProperties(fname=p)
    return None


my_font = get_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
if my_font:
    plt.rcParams['font.sans-serif'] = [my_font.get_name()]

# --- 3. 数据预读取与全局参数计算 ---
china_map = gpd.read_file(shp_path, encoding='UTF-8')
china_map['short_name'] = china_map['name'].astype(str).str[:2]

crop_syrs_dict = {}
global_max_intensity = 0

for crop in crops:
    try:
        df = pd.read_excel(syrs_path, sheet_name=crop, index_col=0)
        df.index = df.index.astype(str).str[:2]
        crop_syrs_dict[crop] = df
        intensity = df.where(df < 0).mean(axis=1).abs()
        if intensity.max() > global_max_intensity:
            global_max_intensity = intensity.max()
    except Exception as e:
        print(f"读取 {crop} 数据失败: {e}")

v_max = min(global_max_intensity, 1.5)

# --- 4. 开始绘图 ---
fig, axes = plt.subplots(2, 3, figsize=(24, 13), dpi=300, gridspec_kw={'height_ratios': [1.2, 1]})

for i, crop in enumerate(crops):
    # --- A. 第一行：空间损失强度地图 ---
    ax_map = axes[0, i]
    try:
        df_syrs = crop_syrs_dict[crop]
        loss_intensity = df_syrs.where(df_syrs < 0).mean(axis=1).abs()
        province_data = loss_intensity.reset_index().rename(columns={0: 'intensity', 'index': 'short_name'})

        merged = china_map.merge(province_data, on='short_name', how='left')

        china_map.plot(ax=ax_map, color='#f5f5f5', edgecolor='white', linewidth=0.5)
        merged.plot(
            column='intensity', cmap='YlOrRd', vmin=0, vmax=v_max,
            ax=ax_map, edgecolor='0.4', linewidth=0.6,
            legend=True,
            legend_kwds={
                'label': "平均减产强度 (|σ|)",
                'orientation': "horizontal",
                'shrink': 0.6,
                'pad': 0.02,
                'aspect': 35
            }
        )
        ax_map.set_title(f'({chr(97 + i)}) {crop}减产强度空间格局', fontsize=22, pad=10)
        ax_map.set_axis_off()

        # --- B. 第二行：累积减产压力堆叠面积图 (Terra 配色版) ---
        ax_area = axes[1, i]
        top5_provs = loss_intensity.sort_values(ascending=False).head(5).index.tolist()
        years = df_syrs.columns.astype(str).str.extract('(\d+)')[0].astype(int).values

        plot_data = df_syrs.loc[top5_provs].copy()
        # 处理减产绝对值用于堆叠
        stacked_values = plot_data.apply(lambda row: row.map(lambda x: abs(x) if x < 0 else 0), axis=1).values

        # 1. 绘制面积堆叠 (使用原来的高级色)
        ax_area.stackplot(years, stacked_values,
                          labels=top5_provs,
                          colors=colors_list,
                          alpha=0.85,
                          edgecolor='white',
                          linewidth=0.3,
                          zorder=2)

        # 2. 绘制总压力包络线 (让脊线更清晰)
        total_pressure = stacked_values.sum(axis=0)
        ax_area.plot(years, total_pressure, color='#2c3e50', lw=2.2, ls='-', label='总累积压力', zorder=5)

        # 3. 细节美化
        ax_area.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_area.grid(axis='y', ls='--', alpha=0.3)
        ax_area.spines['top'].set_visible(False)
        ax_area.spines['right'].set_visible(False)

        # 4. 极端灾年警戒线 (2.0σ)
        ax_area.axhline(2.0, color='#e74c3c', lw=1, ls='--', alpha=0.5, zorder=1)

        ax_area.set_title(f'({chr(100 + i)}) {crop}核心区累积减产压力演变', fontsize=20, pad=15)
        ax_area.tick_params(labelsize=12)

        if i == 0:
            ax_area.set_ylabel('累积损失强度 (Σ|SYRS|)', fontsize=16)
        ax_area.set_xlabel('年份', fontsize=14)

        ax_area.legend(loc='upper left', fontsize=10, ncol=2, frameon=False)
        ax_area.set_ylim(0, max(total_pressure.max() * 1.3, 3.0))

    except Exception as e:
        print(f"绘制 {crop} 详情图时出错: {e}")

# --- 5. 整体排版调整 ---
if my_font:
    plt.suptitle('中国主要粮食作物减产强度时空分异特征 (2002–2022)',
                 fontsize=30, y=0.98)

plt.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.94, hspace=0.35, wspace=0.2)

plt.savefig(save_map_path, bbox_inches='tight')
plt.show()