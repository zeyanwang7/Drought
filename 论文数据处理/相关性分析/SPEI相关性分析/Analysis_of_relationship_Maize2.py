import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Path and Parameter Settings ---
spei_corr_path = r'E:\data\计算结果\相关性分析\Maize_SPEI_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析'

# Define the growing season (March to December)
START_PERIOD = 9
END_PERIOD = 46

# --- 2. Data Processing ---
if not os.path.exists(spei_corr_path):
    print(f"❌ Error: File not found at {spei_corr_path}")
else:
    df_res = pd.read_csv(spei_corr_path)
    total_provinces = df_res['Province'].nunique()
    print(f"📊 Total number of provinces analyzed: {total_provinces}")

    # --- 3. Calculate Significant Positive Correlation (%) ---
    stat_list = []
    for p in range(START_PERIOD, END_PERIOD + 1):
        period_data = df_res[df_res['Period'] == p]

        # Filter: Positive R and P-value < 0.05
        pos_sig_data = period_data[(period_data['R'] > 0) & (period_data['P_Value'] < 0.05)]
        pos_pct = (len(pos_sig_data) / total_provinces) * 100

        stat_list.append({
            'Period': p,
            'Positive_Pct': pos_pct
        })

    df_plot = pd.DataFrame(stat_list)

    # --- 4. Plotting (Academic Style for Journals) ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Plotting the main line
    ax.plot(df_plot['Period'], df_plot['Positive_Pct'],
            color='#d73027', linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', label='Sig. Positive Proportion (%)')

    # Fill area under the curve
    #ax.fill_between(df_plot['Period'], df_plot['Positive_Pct'], color='#d73027', alpha=0.1)

    # --- 5. Aesthetics and Labels ---
    ax.set_title('Evolution of Yield Sensitivity to SPEI for Maize (Mar-Dec)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month / Growth Period', fontsize=12)
    ax.set_ylabel('Proportion of Significant Provinces (%)', fontsize=12)

    # Set Y-axis limits
    max_val = df_plot['Positive_Pct'].max()
    ax.set_ylim(0, max_val + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Map Period to Month Names
    month_map = {9: 'Mar', 13: 'Apr', 17: 'May', 21: 'Jun', 25: 'Jul', 29: 'Aug', 33: 'Sep', 37: 'Oct', 41: 'Nov',
                 45: 'Dec'}
    ax.set_xticks(list(month_map.keys()))
    ax.set_xticklabels(list(month_map.values()))

    # --- 已删除灰色矩形代码 (ax.axvspan) ---

    # Annotate the peak
    # peak_row = df_plot.loc[df_plot['Positive_Pct'].idxmax()]
    # ax.annotate(f'Peak: {peak_row["Positive_Pct"]:.1f}%',
    #             xy=(peak_row['Period'], peak_row['Positive_Pct']),
    #             xytext=(peak_row['Period'] + 1.5, peak_row['Positive_Pct'] + 5),
    #             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    #             fontsize=11, fontweight='bold')

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    # --- 6. Export ---
    save_path = os.path.join(output_dir, 'Maize_Drought_Sensitivity_Proportion_EN_Final.png')
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()

    print(f"✅ Final Version (No Shaded Area) Exported Successfully!")