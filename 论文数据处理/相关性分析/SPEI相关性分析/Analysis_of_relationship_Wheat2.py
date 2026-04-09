import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Path and Parameter Settings ---
spei_corr_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析'

# --- 2. Data Processing ---
if not os.path.exists(spei_corr_path):
    print(f"❌ Error: File not found at {spei_corr_path}")
else:
    df_res = pd.read_csv(spei_corr_path)
    total_provinces = df_res['Province'].nunique()

    # --- 修改点 1: 扩展序列至次年 9 月 ---
    # Oct-Dec: 37-48
    # Jan-Sep: 1-36 (每个月 4 个 Period)
    oct_dec = list(range(37, 48 + 1))
    jan_sep = list(range(1, 36 + 1))  # 扩展到 36 (9月底)

    custom_period_order = oct_dec + jan_sep

    # --- 3. Calculate Significant Positive Correlation (%) ---
    stat_list = []
    for p in custom_period_order:
        period_data = df_res[df_res['Period'] == p]
        if period_data.empty:
            continue

        pos_sig_data = period_data[(period_data['R'] > 0) & (period_data['P_Value'] < 0.05)]
        pos_pct = (len(pos_sig_data) / total_provinces) * 100

        stat_list.append({
            'Period': p,
            'Positive_Pct': pos_pct
        })

    df_plot = pd.DataFrame(stat_list)

    # --- 4. Plotting ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300) # 进一步加宽画布

    plot_indices = range(len(df_plot))

    ax.plot(plot_indices, df_plot['Positive_Pct'],
            color='#d73027', linewidth=2.5, marker='o',
            markersize=4, markerfacecolor='white', label='Sig. Positive Proportion (%)')

    # --- 5. Aesthetics and Labels ---
    ax.set_title('Yield Sensitivity to SPEI: Oct to Subsequent Sep for Wheat', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month (Cross-year Cycle)', fontsize=12)
    ax.set_ylabel('Proportion of Significant Provinces (%)', fontsize=12)

    ax.set_ylim(0, df_plot['Positive_Pct'].max() + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- 修改点 2: 映射 9 月标签 ---
    month_labels = {
        0: 'Oct', 4: 'Nov', 8: 'Dec',
        12: 'Jan', 16: 'Feb', 20: 'Mar',
        24: 'Apr', 28: 'May', 32: 'Jun',
        36: 'Jul', 40: 'Aug', 44: 'Sep'  # 新增 9月
    }

    current_ticks = [k for k in month_labels.keys() if k < len(df_plot)]
    ax.set_xticks(current_ticks)
    ax.set_xticklabels([month_labels[k] for k in current_ticks])

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    # --- 6. Export ---
    save_path = os.path.join(output_dir, 'Wheat_Sensitivity_Oct_to_Sep_EN.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"✅ Successfully exported: Oct to subsequent Sep analysis.")