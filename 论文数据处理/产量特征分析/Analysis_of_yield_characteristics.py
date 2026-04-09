import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import re

# ================= 1. 环境与路径配置 =================
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 输入文件
yield_unit_path = r'E:\data\crop_yield\各省年度单产_排序版.xlsx'  # 单产数据
yield_total_path = r'E:\data\yeild_data\作物产量数据2000-2022_已排序.xlsx'  # 总产数据
shp_path = r'E:\数据集\中国_省\中国_省2.shp'

# 输出文件
syrs_output_path = r'E:\data\计算结果\表格\单产标准化残差_SYRS.xlsx'
save_path_fig = r'E:\data\计算结果\3.1.1_单产损失特征图_总产加权版.png'

crops = ['小麦', '玉米', '水稻']
years = [int(y) for y in range(2002, 2023)]


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


def calculate_syrs(df):
    """二次项去趋势并标准化"""
    syrs_df = pd.DataFrame(index=df.index, columns=df.columns)
    X = np.array(years).reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    for province in df.index:
        y = df.loc[province].values.astype(float)
        mask = ~np.isnan(y)
        if not mask.any(): continue

        model = LinearRegression().fit(X_poly[mask], y[mask])
        trend = model.predict(X_poly)
        residuals = y - trend
        # 标准化
        syrs = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)
        syrs_df.loc[province] = syrs
    return syrs_df.astype(float)


# ================= 3. 执行计算与绘图 =================
def main_analysis():
    print("正在处理地理数据...")
    try:
        china_map = gpd.read_file(shp_path, encoding='utf-8')
        china_map['match_name'] = china_map['name'].apply(clean_name)
        temp_map = china_map.to_crs(epsg=3857)  # 投影计算面积
        china_map['area_m2'] = temp_map.geometry.area
    except Exception as e:
        print(f"地图加载失败: {e}")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 20), gridspec_kw={'width_ratios': [1, 1.2]})

    with pd.ExcelWriter(syrs_output_path) as writer:
        for i, crop in enumerate(crops):
            print(f"--- 正在处理: {crop} ---")

            # 1. 计算单产 SYRS
            df_unit = pd.read_excel(yield_unit_path, sheet_name=crop, index_col=0)
            df_unit.index = df_unit.index.map(clean_name)
            df_syrs = calculate_syrs(df_unit)
            df_syrs.to_excel(writer, sheet_name=crop)  # 保存结果

            # 2. 读取总产量并计算各省权重 (基于2002-2022平均总产)
            df_total = pd.read_excel(yield_total_path, sheet_name=crop, index_col=0)
            df_total.index = df_total.index.map(clean_name)
            # 确保只取 2002-2022 的年份列进行权重计算
            cols = [c for c in df_total.columns if str(c) in [str(y) for y in years]]
            avg_production = df_total[cols].mean(axis=1)
            weights = avg_production / avg_production.sum()

            # --- A. 空间图 (左图) ---
            spatial_risk = df_syrs[df_syrs < 0].mean(axis=1)
            plot_map = china_map.merge(spatial_risk.rename('risk'), left_on='match_name', right_index=True, how='left')
            plot_map.loc[plot_map['area_m2'] < 1e8, 'risk'] = np.nan  # 过滤南海

            ax_map = axes[i, 0]
            plot_map.plot(column='risk', cmap='RdYlGn', legend=True,
                          legend_kwds={'label': "减产强度 (SYRS < 0)", 'orientation': "horizontal", 'shrink': 0.6},
                          ax=ax_map, missing_kwds={"color": "#f2f2f2", "edgecolor": "#d9d9d9", "linewidth": 0.5})
            ax_map.set_title(f"({chr(97 + i * 2)}) {crop}单产损失空间分布", fontsize=15, fontweight='bold')
            ax_map.axis('off')

            # --- B. 时间图 (右图：总产加权) ---
            # 这里的加权：各省单产残差 * 各省总产量权重
            weighted_syrs = (df_syrs.multiply(weights, axis=0)).sum(axis=0)

            ax_line = axes[i, 1]
            y_values = weighted_syrs.values
            bar_colors = ['#d63031' if v < 0 else '#00b894' for v in y_values]

            ax_line.bar(years, y_values, color=bar_colors, alpha=0.3, width=0.7)
            ax_line.plot(years, y_values, marker='o', color='#2d3436', linewidth=1.5, markersize=5,
                         label='全国总产加权SYRS')
            ax_line.axhline(0, color='black', linewidth=0.8)
            ax_line.axhline(-1, color='#e17055', linestyle='--', alpha=0.7, label='显著波动阈值')

            ax_line.set_title(f"({chr(98 + i * 2)}) {crop}全国加权单产波动序列", fontsize=15, fontweight='bold')
            ax_line.set_xticks(range(2002, 2023, 2))
            ax_line.set_ylim(-2.5, 2.5)
            ax_line.set_ylabel("Weighted SYRS")
            ax_line.grid(axis='y', linestyle=':', alpha=0.5)
            if i == 0: ax_line.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(save_path_fig, dpi=300, bbox_inches='tight')
    print(f"\n[完成] SYRS 表格已保存至: {syrs_output_path}")
    print(f"[完成] 3x2 特征图已保存至: {save_path_fig}")
    plt.show()


if __name__ == "__main__":
    main_analysis()