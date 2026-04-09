import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# --- 0. 环境与字体设置 ---
warnings.filterwarnings("ignore")


def set_ch_font():
    paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']
    for p in paths:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=p).get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False


set_ch_font()


# --- 1. 二次型去趋势函数 ---
def calculate_detrended_cv(series):
    """使用二次多项式拟合去趋势并计算变异系数"""
    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    # 构建二次特征: y = ax^2 + bx + c
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)

    # 线性回归拟合趋势线
    model = LinearRegression()
    model.fit(x_poly, y)
    y_trend = model.predict(x_poly)

    # 计算残差并恢复均值基准
    y_detrended = (y - y_trend) + y.mean()

    # 计算去趋势后的 CV
    cv_detrended = np.std(y_detrended) / np.mean(y_detrended)
    return cv_detrended


# --- 2. 路径与配置 ---
syrs_path = r'E:\data\计算结果\表格\单产标准化残差_SYRS.xlsx'
yield_path = r'E:\data\crop_yield\各省年度单产_排序版.xlsx'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
crops = ['小麦', '玉米', '水稻']
quad_colors = {
    '高强度-高波动 (极高风险)': '#d73027',
    '高强度-低波动 (慢性风险)': '#fc8d59',
    '低强度-高波动 (不稳风险)': '#fee090',
    '低强度-低波动 (低风险)': '#91bfdb'
}

# --- 3. 准备地理数据 ---
china_map = gpd.read_file(shp_path, encoding='UTF-8')
china_map['short_name'] = china_map['name'].astype(str).str[:2]

# --- 4. 绘图循环 (3行2列) ---
fig, axes = plt.subplots(3, 2, figsize=(20, 24), dpi=300,
                         gridspec_kw={'width_ratios': [1, 1.2], 'hspace': 0.22, 'wspace': -0.08})

for i, crop in enumerate(crops):
    # A. 数据加载与前两字对齐
    df_syrs = pd.read_excel(syrs_path, sheet_name=crop, index_col=0)
    df_yield = pd.read_excel(yield_path, sheet_name=crop, index_col=0)
    df_syrs.index = df_syrs.index.astype(str).str[:2]
    df_yield.index = df_yield.index.astype(str).str[:2]

    # B. 指标计算
    # X轴：强度 (利用已有的标准化残差SYRS)
    intensity = df_syrs.apply(lambda row: row[row < 0].abs().mean(), axis=1)

    # Y轴：稳定性 (使用二次多项式去趋势后的CV)
    cv_series = df_yield.apply(calculate_detrended_cv, axis=1)

    risk_df = pd.DataFrame({'Int': intensity, 'CV': cv_series}).dropna()
    m_int, m_cv = risk_df['Int'].mean(), risk_df['CV'].mean()


    def classify(row):
        if row['Int'] >= m_int and row['CV'] >= m_cv: return '高强度-高波动 (极高风险)'
        if row['Int'] >= m_int and row['CV'] < m_cv:  return '高强度-低波动 (慢性风险)'
        if row['Int'] < m_int and row['CV'] >= m_cv:  return '低强度-高波动 (不稳风险)'
        return '低强度-低波动 (低风险)'


    risk_df['Quadrant'] = risk_df.apply(classify, axis=1)

    # C. 左侧：散点图 (a, c, e)
    ax_scat = axes[i, 0]
    scat_labels = ['a', 'c', 'e']
    for label, color in quad_colors.items():
        sub = risk_df[risk_df['Quadrant'] == label]
        ax_scat.scatter(sub['Int'], sub['CV'], c=color, s=180, edgecolors='w', alpha=0.9,
                        label=label if i == 0 else "")

    ax_scat.axvline(m_int, color='#666666', ls='--', lw=1, alpha=0.6)
    ax_scat.axhline(m_cv, color='#666666', ls='--', lw=1, alpha=0.6)

    for prov in risk_df.index:
        ax_scat.text(risk_df.loc[prov, 'Int'] + 0.002, risk_df.loc[prov, 'CV'] + 0.0005,
                     prov, fontsize=9, alpha=0.8)

    ax_scat.set_title(f"({scat_labels[i]}) {crop}风险特征分区 (二次去趋势)", loc='left', fontsize=18, fontweight='bold')
    ax_scat.set_xlabel('平均减产强度 (Mean |SYRS|)', fontsize=13)
    ax_scat.set_ylabel('去趋势单产变异系数 (Detrended CV)', fontsize=13)
    if i == 0: ax_scat.legend(loc='upper right', fontsize=10, frameon=True)

    # D. 右侧：地理图 (b, d, f)
    ax_map = axes[i, 1]
    map_labels = ['b', 'd', 'f']
    merged = china_map.merge(risk_df.reset_index().rename(columns={'index': 'short_name'}), on='short_name', how='left')

    china_map.plot(ax=ax_map, color='#f2f2f2', edgecolor='white', linewidth=0.5)
    for label, color in quad_colors.items():
        merged[merged['Quadrant'] == label].plot(ax=ax_map, color=color, edgecolor='0.4', linewidth=0.5)

    ax_map.set_axis_off()
    ax_map.set_title(f"({map_labels[i]}) {crop}风险空间格局", loc='left', fontsize=18, fontweight='bold')

# --- 5. 导出 ---
plt.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.06)
plt.suptitle('基于二次型去趋势的粮食作物单产损失风险评估图谱', fontsize=26, fontweight='bold', y=0.98)
save_path = r'E:\data\计算结果\表格\Crop_Risk_Quadratic_Detrended.png'
plt.savefig(save_path, bbox_inches='tight')
plt.show()