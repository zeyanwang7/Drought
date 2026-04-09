import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 路径与参数设置 ---
# 使用之前生成的 SVPD 相关性结果文件
svpd_corr_path = r'E:\data\计算结果\相关性分析_SVPD\Maize_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析_SVPD'

# 定义水稻的主要生长期（通常为 4 月到 10 月，或者保持 3-12 月以观察全年）
START_PERIOD = 9
END_PERIOD = 46

# --- 2. 数据处理 ---
if not os.path.exists(svpd_corr_path):
    print(f"❌ 错误: 找不到文件 {svpd_corr_path}")
else:
    df_res = pd.read_csv(svpd_corr_path)
    total_provinces = df_res['Province'].nunique()
    print(f"📊 分析的省份总数: {total_provinces}")

    # --- 3. 计算显著负相关比例 (%) ---
    # 注意：SVPD 越高越干，所以我们统计 R < 0 (即干热导致减产) 的比例
    stat_list = []
    for p in range(START_PERIOD, END_PERIOD + 1):
        period_data = df_res[df_res['Period'] == p]

        # 过滤条件：负相关 (R < 0) 且显著 (P < 0.05)
        neg_sig_data = period_data[(period_data['R'] < 0) & (period_data['P_Value'] < 0.05)]
        neg_pct = (len(neg_sig_data) / total_provinces) * 100

        stat_list.append({
            'Period': p,
            'Negative_Pct': neg_pct
        })

    df_plot = pd.DataFrame(stat_list)

    # --- 4. 绘图 (学术期刊风格) ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 绘制主曲线
    # 使用深橙色或红色 (Hot color) 来代表 VPD 干热压力
    ax.plot(df_plot['Period'], df_plot['Negative_Pct'],
            color='#e6550d', linewidth=2.5, marker='s', # 使用方形 marker 区分 SPEI
            markersize=5, markerfacecolor='white', label='Sig. Negative Proportion (VPD Stress %)')

    # --- 5. 美化与标签 ---
    ax.set_title('Evolution of Maize Yield Sensitivity to SVPD (Mar-Dec)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month / Growth Period', fontsize=12)
    ax.set_ylabel('Proportion of Significant Provinces (%)', fontsize=12)

    # 设置 Y 轴范围
    max_val = df_plot['Negative_Pct'].max()
    ax.set_ylim(0, max_val + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 月份映射（8天一期，约每4期一个月）
    month_map = {9: 'Mar', 13: 'Apr', 17: 'May', 21: 'Jun', 25: 'Jul', 29: 'Aug', 33: 'Sep', 37: 'Oct', 41: 'Nov', 45: 'Dec'}
    ax.set_xticks(list(month_map.keys()))
    ax.set_xticklabels(list(month_map.values()))

    # 突出显示夏季（7-8月），这通常是水稻受 VPD 影响最大的窗口期
    # ax.axvspan(25, 32, color='orange', alpha=0.05, label='Critical Summer Period')

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    # --- 6. 导出 ---
    save_path = os.path.join(output_dir, 'Maize_SVPD_Sensitivity_Proportion_Final.png')
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()

    print(f"✅ SVPD 敏感性比例图导出成功！")
    print(f"📍 保存路径: {save_path}")