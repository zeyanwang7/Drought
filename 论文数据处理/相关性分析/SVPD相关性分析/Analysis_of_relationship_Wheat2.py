import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 路径与参数设置 ---
# 使用之前生成的 SVPD 相关性结果文件
svpd_corr_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析_SVPD'

# --- 2. 数据处理 ---
if not os.path.exists(svpd_corr_path):
    print(f"❌ Error: File not found at {svpd_corr_path}")
else:
    df_res = pd.read_csv(svpd_corr_path)
    total_provinces = df_res['Province'].nunique()
    print(f"📊 Total number of provinces analyzed: {total_provinces}")

    # --- 时间序列定义: 跨年周期 (Oct to subsequent Sep) ---
    # 当年 Oct-Dec: 37-46 (注意: 一年通常定义为 46 个 8天周期)
    # 次年 Jan-Sep: 1-36
    oct_dec = list(range(37, 46 + 1))
    jan_sep = list(range(1, 36 + 1))

    custom_period_order = oct_dec + jan_sep

    # --- 3. 计算显著负相关比例 (%) ---
    # SVPD 越高代表干热胁迫越严重，因此统计 R < 0 且 P < 0.05 的比例
    stat_list = []
    for p in custom_period_order:
        period_data = df_res[df_res['Period'] == p]
        if period_data.empty:
            continue

        # 统计负相关且显著的省份数量
        neg_sig_data = period_data[(period_data['R'] < 0) & (period_data['P_Value'] < 0.05)]
        neg_pct = (len(neg_sig_data) / total_provinces) * 100

        stat_list.append({
            'Period': p,
            'Negative_Pct': neg_pct
        })

    df_plot = pd.DataFrame(stat_list)

    # --- 4. 绘图配置 (学术期刊风格) ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    # plot_indices 用于在 X 轴上平铺跨年数据
    plot_indices = range(len(df_plot))

    # 使用橙红色 (#e6550d) 代表大气干热压力
    ax.plot(plot_indices, df_plot['Negative_Pct'],
            color='#e6550d', linewidth=2.5, marker='s', # 使用方形 marker 区分
            markersize=5, markerfacecolor='white', label='Sig. Negative Proportion (VPD Stress %)')

    # 填充曲线下方，增加视觉效果
    ax.fill_between(plot_indices, df_plot['Negative_Pct'], color='#e6550d', alpha=0.1)

    # --- 5. 美化与标签 ---
    ax.set_title('Wheat Yield Sensitivity to SVPD: Oct to Subsequent Sep', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month (Growing Season Cycle)', fontsize=12)
    ax.set_ylabel('Proportion of Significant Provinces (%)', fontsize=12)

    # 设置 Y 轴范围
    max_val = df_plot['Negative_Pct'].max()
    ax.set_ylim(0, max_val + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- 映射月份标签 ---
    # 逻辑: oct_dec 长度约 10 期，然后接 jan_sep
    # 我们根据 plot_indices 的位置设置刻度
    month_labels = {
        0: 'Oct', 4: 'Nov', 8: 'Dec',
        12: 'Jan', 16: 'Feb', 20: 'Mar',
        24: 'Apr', 28: 'May', 32: 'Jun',
        36: 'Jul', 40: 'Aug', 44: 'Sep'
    }

    # 过滤掉超出索引范围的标签
    current_ticks = [k for k in month_labels.keys() if k < len(df_plot)]
    ax.set_xticks(current_ticks)
    ax.set_xticklabels([month_labels[k] for k in current_ticks])

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    # --- 6. 导出 ---
    save_path = os.path.join(output_dir, 'Wheat_SVPD_Sensitivity_Oct_to_Sep.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"✅ Successfully exported SVPD sensitivity plot: {save_path}")