import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 路径与字体设置 ---
# 确保文件路径指向棉花的相关性结果
spei_corr_path = r'E:\data\计算结果\相关性分析\Cotton_SPEI_Yield_Correlation.csv'
svpd_corr_path = r'E:\data\计算结果\相关性分析_SVPD\Cotton_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析_对比'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 解决中文显示与负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'MicroSoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义时间轴 (4月下至10月上) ---
# 4月下（15期）到10月上（37期），共23个 8-day 周期
custom_period_order = list(range(15, 37 + 1))

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
            # 显著正相关：土壤水分（SPEI）增加，产量增加 -> 水分限制
            sig_count = len(period_data[(period_data['R'] > 0) & (period_data['P_Value'] < 0.05)])
        else:
            # 显著负相关：干热程度（SVPD）增加，产量下降 -> 大气干热胁迫
            sig_count = len(period_data[(period_data['R'] < 0) & (period_data['P_Value'] < 0.05)])

        pct_results.append((sig_count / total_provinces) * 100)
    return pct_results

# --- 4. 执行数据提取 ---
print("正在提取并计算 4月-10月 棉花显著性比例...")
spei_pcts = get_significant_percentage(spei_corr_path, direction='positive')
svpd_pcts = get_significant_percentage(svpd_corr_path, direction='negative')

# --- 5. 绘图展示 ---
print("正在生成对比曲线图...")
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

x_axis = np.arange(len(custom_period_order))

# 绘制曲线
ax.plot(x_axis, spei_pcts, color='#0571b0', linewidth=2.5,
        marker='o', markersize=5, markerfacecolor='white', label='SPEI 显著正相关占比 (水分限制)')

ax.plot(x_axis, svpd_pcts, color='#e6550d', linewidth=2.5,
        marker='s', markersize=5, markerfacecolor='white', label='SVPD 显著负相关占比 (干热胁迫)')

# --- 6. 坐标轴修饰 (纯月份标签) ---
ax.set_title('棉花产量对 SPEI 与 SVPD 的敏感性演变 (4月-10月)', fontsize=18, fontweight='bold', pad=25)
ax.set_xlabel('生长阶段 (月份)', fontsize=13)
ax.set_ylabel('显著相关省份占比 (%)', fontsize=13)

# 映射逻辑（基于15期开始）：
# 0->4月下, 4->5月, 8->6月, 12->7月, 16->8月, 20->9月, 22->10月
month_labels = {0: '4月', 4: '5月', 8: '6月', 12: '7月', 16: '8月', 20: '9月', 22: '10月'}
ax.set_xticks(list(month_labels.keys()))
ax.set_xticklabels(list(month_labels.values()))

# --- 7. 添加棉花关键生育时段阴影 ---
# 棉花现蕾期：通常为6月中旬至7月上旬
# 对应 x_axis 索引约为 8 到 12
ax.axvspan(8, 12, color='green', alpha=0.1, label='现蕾期')

# 棉花花铃期：棉花需水高峰与干热最敏感期，通常为7月中旬至8月下旬
# 对应 x_axis 索引约为 12 到 18
ax.axvspan(12, 18, color='red', alpha=0.1, label='花铃期 (敏感高峰)')

# 棉花吐絮期：9月至10月
ax.axvspan(20, 22, color='gray', alpha=0.1, label='吐絮期')

# 设置 Y 轴范围并添加网格
max_val = max(max(spei_pcts), max(svpd_pcts)) if (spei_pcts or svpd_pcts) else 50
ax.set_ylim(0, max_val + 15)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 图例设置
ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, ncol=2)

# 完善布局并保存
plt.tight_layout()
save_path = os.path.join(output_dir, 'Cotton_SPEI_SVPD_Sensitivity_Comparison.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘图成功！已标注棉花现蕾、花铃及吐絮关键时段。文件保存至: {save_path}")