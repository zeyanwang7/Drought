import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 路径与参数设置 ---
yield_excel_path = r'E:\data\yeild_data\作物产量数据2000-2022_已排序.xlsx'
spei_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
svpd_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'

crop_name = '小麦'
target_periods = list(range(12, 29))
# 月份映射字典
month_labels = {12: '3月下', 14: '4月上', 17: '4月下', 20: '5月中', 23: '6月上', 26: '6月下', 28: '7月中'}

# --- 2. 产量权重处理 ---
df_yield = pd.read_excel(yield_excel_path, sheet_name=crop_name)
province_col = df_yield.columns[0]
# 计算各省2000-2022平均产量
df_yield['Avg_Yield'] = df_yield.iloc[:, 1:].mean(axis=1)
province_weight = df_yield[[province_col, 'Avg_Yield']].copy()
province_weight.columns = ['Province', 'Avg_Yield']
# 统一省份名称格式
province_weight['Province'] = province_weight['Province'].astype(str).str.replace(r'(省|自治区|市|维吾尔|壮族|回族)', '', regex=True).str.strip()

# --- 3. 计算加权相关性函数 ---
def get_weighted_r(filepath, weight_df):
    if not os.path.exists(filepath): return None
    df_corr = pd.read_csv(filepath)
    df_corr['Province'] = df_corr['Province'].astype(str).str.replace(r'(省|自治区|市|维吾尔|壮族|回族)', '', regex=True).str.strip()
    merged = pd.merge(df_corr, weight_df, on='Province', how='inner')
    # 计算 |R| * 产量权重
    merged['W_R'] = merged['R'].abs() * merged['Avg_Yield']
    res = merged.groupby('Period').agg({'W_R': 'sum', 'Avg_Yield': 'sum'})
    return res['W_R'] / res['Avg_Yield']

# 执行计算
weighted_spei = get_weighted_r(spei_path, province_weight)
weighted_svpd = get_weighted_r(svpd_path, province_weight)

# 构建绘图数据
df_plot = pd.DataFrame({'SPEI': weighted_spei, 'SVPD': weighted_svpd}).loc[target_periods].dropna()
df_plot['Envelope'] = df_plot.max(axis=1)

# --- 4. 绘图实现 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布，增加高度预留给双轴和图例
fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)

# --- 修正后的物候背景设置 ---
# 冬小麦关键期 (拔节-灌浆): 4月上旬(P13) 到 5月下旬(P20)
ax1.axvspan(13, 20, color='#1f77b4', alpha=0.15, label='冬小麦敏感期 (P13-P20)')

# 春小麦关键期 (拔节-扬花): 6月上旬(P21) 到 7月上旬(P26)
ax1.axvspan(21, 26, color='#ff7f0e', alpha=0.15, label='春小麦敏感期 (P21-P26)')

# B. 绘制主要折线
ax1.plot(df_plot.index, df_plot['SPEI'], label='SPEI (产量加权强度)',
         color='#1f77b4', linestyle='--', marker='o', alpha=0.7, markersize=6)
ax1.plot(df_plot.index, df_plot['SVPD'], label='SVPD (产量加权强度)',
         color='#d62728', linestyle='--', marker='s', alpha=0.7, markersize=6)
ax1.plot(df_plot.index, df_plot['Envelope'], label='最大监测能力包络线',
         color='black', linewidth=3, zorder=10)

# C. 阴影填充与优势标注
# ax1.fill_between(df_plot.index, df_plot['Envelope'], color='gray', alpha=0.1)
for p in df_plot.index:
    tag = 'SPEI' if df_plot.loc[p, 'SPEI'] > df_plot.loc[p, 'SVPD'] else 'SVPD'
    ax1.text(p, df_plot.loc[p, 'Envelope'] + 0.015, tag, ha='center', fontweight='bold', size=8)

# D. 底部 X 轴 (Period)
ax1.set_xlabel('生育阶段 (Period)', fontsize=12, labelpad=10)
ax1.set_xticks(df_plot.index)
ax1.set_ylabel('加权平均相关性强度 |Rw|', fontsize=12)
ax1.grid(axis='y', linestyle=':', alpha=0.4)
# 调整 Y 轴范围给顶部留白
ax1.set_ylim(df_plot[['SPEI','SVPD']].values.min()*0.8, df_plot['Envelope'].max()*1.3)

# E. 顶部 X 轴 (Month)
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(list(month_labels.keys()))
ax2.set_xticklabels(list(month_labels.values()), fontsize=11, fontweight='bold')

# F. 标题与图例 (精细排版防止重叠)
plt.title(f'{crop_name}产量加权相关性包络分析 (冬春小麦物候对照图)', fontsize=16, fontweight='bold', y=1.26)

# 获取所有图例句柄并合并
handles, labels = ax1.get_legend_handles_labels()
# 图例放在标题下方，采用 3 列排列
ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.22),
           ncol=3, frameon=False, fontsize=10, columnspacing=1.2)

# G. 调整布局，收缩绘图区高度，将顶部空间留给标题/图例/双轴
plt.subplots_adjust(top=0.78, bottom=0.12, left=0.1, right=0.95)

# 保存
output_folder = r'E:\data\计算结果'
if not os.path.exists(output_folder): os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, 'Wheat_Final_Weighted_Envelope_Professional.png'), bbox_inches='tight')

plt.show()

print(f"✅ 带有深色重叠区和双X轴的小麦包络图已生成！")