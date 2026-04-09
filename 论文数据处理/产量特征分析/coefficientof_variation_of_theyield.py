import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

# ================= 1. 环境与路径配置 =================
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 输入：你最新保存的 SYRS 结果表格
syrs_input_path = r'E:\data\计算结果\表格\单产标准化残差_SYRS.xlsx'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
# 输出：变异系数地图
save_path_cv = r'E:\data\计算结果\3.1.1_单产波动强度地图.png'

crops = ['小麦', '玉米', '水稻']


# ================= 2. 核心函数 =================
def clean_name(name):
    if pd.isna(name): return ""
    name = str(name).strip()
    name = re.sub(r'(省|市|自治区|回族|壮族|维吾尔|特别行政区)', '', name)
    if name == '内蒙': name = '内蒙古'
    special_cases = {'广西壮族': '广西', '新疆维吾尔': '新疆', '宁夏回族': '宁夏'}
    for k, v in special_cases.items():
        if k in name: name = v
    return name


# ================= 3. 绘图主程序 =================
def plot_cv_analysis():
    print("正在生成单产波动强度（变异系数）地图...")

    try:
        # 加载地图并预处理
        china_map = gpd.read_file(shp_path, encoding='utf-8')
        china_map['match_name'] = china_map['name'].apply(clean_name)
        # 投影计算面积，用于过滤南海岛礁
        temp_map = china_map.to_crs(epsg=3857)
        china_map['area_m2'] = temp_map.geometry.area
    except Exception as e:
        print(f"地图加载失败: {e}")
        return

    # 1行3列布局
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))

    for i, crop in enumerate(crops):
        print(f"分析中: {crop}")
        try:
            # 读取 SYRS 数据
            df_syrs = pd.read_excel(syrs_input_path, sheet_name=crop, index_col=0)
            df_syrs.index = df_syrs.index.map(clean_name)

            # 计算波动强度（SYRS 的标准差）
            # 在标准化序列中，标准差能够完美表征变异特征
            cv_values = df_syrs.std(axis=1)

            # 合并地理数据
            plot_map = china_map.merge(cv_values.rename('cv'), left_on='match_name', right_index=True, how='left')

            # 过滤掉南海微小岛礁
            plot_map.loc[plot_map['area_m2'] < 1e8, 'cv'] = np.nan

            ax = axes[i]
            # 使用 'YlOrRd' (黄-橙-红) 渐变色，深红代表高风险/高波动
            plot_map.plot(column='cv', cmap='YlOrRd', legend=True,
                          legend_kwds={'label': "波动强度 (Std of SYRS)",
                                       'orientation': "horizontal",
                                       'shrink': 0.7, 'pad': 0.05},
                          ax=ax,
                          missing_kwds={"color": "#f5f5f5", "edgecolor": "#d9d9d9", "linewidth": 0.4})

            ax.set_title(f"({chr(97 + i)}) {crop}单产波动空间特征", fontsize=18, fontweight='bold')
            ax.axis('off')

        except Exception as e:
            print(f"处理 {crop} 失败: {e}")

    plt.tight_layout()
    plt.savefig(save_path_cv, dpi=300, bbox_inches='tight')
    print(f"\n[成功] 变异系数地图已生成！保存路径: {save_path_cv}")
    plt.show()


if __name__ == "__main__":
    plot_cv_analysis()