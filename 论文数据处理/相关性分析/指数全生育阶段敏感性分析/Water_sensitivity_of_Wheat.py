import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 路径与字体设置 ---
spei_corr_path = r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv'
svpd_corr_path = r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析_对比'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'MicroSoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义跨年时间轴 (上年9月中至次年7月下) ---
# 上年：34期(9月中) 到 46期(12月底)
last_year_periods = list(range(34, 46 + 1))
# 次年：1期(1月初) 到 27期(7月下)
this_year_periods = list(range(1, 27 + 1))

custom_period_order = last_year_periods + this_year_periods

# --- 3. 计算显著性占比函数 ---
def get_significant_percentage(file_path, direction='positive'):
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到文件 {file_path}")
        return [0] * len(custom_period_order)

    df = pd.read_csv(file_path)
    total_provinces = df['Province'].nunique()

    pct_results = []
    for p in custom_period_order:
        period_data = df[df['Period'] == p]
        if period_data.empty:
            pct_results.append(0)
            continue

        if direction == 'positive':
            # SPEI: 正显著比例
            sig_count = len(period_data[(period_data['R'] > 0) & (period_data['P_Value'] < 0.05)])
        else:
            # SVPD: 负显著比例
            sig_count = len(period_data[(period_data['R'] < 0) & (period_data['P_Value'] < 0.05)])

        pct_results.append((sig_count / total_provinces) * 100)
    return pct_results

# --- 4. 执行数据提取 ---
print("正在提取小麦全生育期显著性比例...")
spei_pcts = get_significant_percentage(spei_corr_path, direction='positive')
svpd_pcts = get_significant_percentage(svpd_corr_path, direction='negative')

# --- 5. 绘图展示 ---
fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
x_axis = np.arange(len(custom_period_order))

# 绘制曲线
ax.plot(x_axis, spei_pcts, color='#0571b0', linewidth=2, marker='o',
        markersize=4, markerfacecolor='white', label='SPEI 显著正相关占比 ')
ax.plot(x_axis, svpd_pcts, color='#e6550d', linewidth=2, marker='s',
        markersize=4, markerfacecolor='white', label='SVPD 显著负相关占比 ')

# --- 6. 坐标轴修饰 (仅保留月份) ---
ax.set_title('小麦产量对 SPEI 与 SVPD 的敏感性演变 (上年9月-次年7月)', fontsize=18, fontweight='bold', pad=25)
ax.set_ylabel('显著相关省份占比 (%)', fontsize=13)
ax.set_xlabel('生长阶段（月份）', fontsize=13, labelpad=15)
# 映射月份标签 (对齐到每月的起始索引)
month_labels = {
    0: '上年9月', 4: '10月', 8: '11月', 12: '12月',
    16: '1月', 20: '2月', 24: '3月', 28: '4月', 32: '5月', 36: '6月', 40: '7月'
}
ax.set_xticks(list(month_labels.keys()))
ax.set_xticklabels(list(month_labels.values()))

# --- 7. 添加冬/春小麦关键发育期阴影 ---
# 冬小麦核心关键期 (抽穗-灌浆)：4月中旬-5月下旬
# 这是北方冬小麦对 SVPD (高温干热) 和 SPEI (水分) 最敏感的时期
ax.axvspan(26, 32, color='red', alpha=0.12, label='冬小麦关键期 (抽穗-灌浆)')

# 春小麦核心关键期 (拔节-抽穗)：6月上旬-7月上旬
# 对应西北、东北春小麦的水分临界期
ax.axvspan(33, 37, color='green', alpha=0.12, label='春小麦关键期 (拔节-扬花)')

# (可选) 增加一个冬小麦越冬期标注，解释为什么前期相关性占比较低
# ax.axvspan(12, 22, color='gray', alpha=0.05, label='越冬期')

# 设置 Y 轴
max_y = max(max(spei_pcts), max(svpd_pcts)) if (spei_pcts or svpd_pcts) else 50
ax.set_ylim(0, max_y + 10)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 图例设置 (分两列显示更整齐)
ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, ncol=2)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Wheat_Lifecycle_Sensitivity_Comparison.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘图成功！时间范围：上年9月中至次年7月下。文件保存至: {save_path}")