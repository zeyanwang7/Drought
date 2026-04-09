import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 路径与字体设置 ---
spei_corr_path = r'E:\data\计算结果\相关性分析\Rice_SPEI_Yield_Correlation.csv'
svpd_corr_path = r'E:\data\计算结果\相关性分析_SVPD\Rice_SVPD_Yield_Correlation.csv'
output_dir = r'E:\data\计算结果\相关性分析_对比'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'MicroSoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 定义时间轴 (3月底至11月) ---
# 第11期约为3月下旬，第41期约为11月中旬
custom_period_order = list(range(11, 41 + 1))

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
            sig_count = len(period_data[(period_data['R'] > 0) & (period_data['P_Value'] < 0.05)])
        else:
            sig_count = len(period_data[(period_data['R'] < 0) & (period_data['P_Value'] < 0.05)])

        pct_results.append((sig_count / total_provinces) * 100)
    return pct_results

# --- 4. 执行数据提取 ---
print("正在提取早/中/晚稻生育期显著性比例...")
spei_pcts = get_significant_percentage(spei_corr_path, direction='positive')
svpd_pcts = get_significant_percentage(svpd_corr_path, direction='negative')

# --- 5. 绘图展示 ---
fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
x_axis = np.arange(len(custom_period_order))

# 绘制曲线
ax.plot(x_axis, spei_pcts, color='#0571b0', linewidth=2.5, marker='o',
        markersize=5, markerfacecolor='white', label='SPEI 显著正相关占比')
ax.plot(x_axis, svpd_pcts, color='#e6550d', linewidth=2.5, marker='s',
        markersize=5, markerfacecolor='white', label='SVPD 显著负相关占比')

# --- 6. 坐标轴修饰 (纯月份标签) ---
ax.set_title('水稻产量对 SPEI 与 SVPD 的敏感性演变 (3月-11月)', fontsize=18, fontweight='bold', pad=25)
ax.set_ylabel('显著相关省份占比 (%)', fontsize=13)
ax.set_xlabel('生长阶段 (月份)', fontsize=13)

# 映射月份标签
month_labels = {
    0: '3月', 4: '4月', 8: '5月', 12: '6月', 16: '7月',
    20: '8月', 24: '9月', 28: '10月', 30: '11月'
}
ax.set_xticks(list(month_labels.keys()))
ax.set_xticklabels(list(month_labels.values()))

# --- 7. 修正后的水稻各类型关键发育期阴影 ---

# 早稻 (抽穗-灌浆期)：6月中旬-7月上旬
# 此阶段 SVPD 显著占比升高通常对应“小暑”前后的高温影响
ax.axvspan(11, 14, color='blue', alpha=0.1, label='早稻关键期 (抽穗-灌浆)')

# 中稻/一季稻 (抽穗-扬花期)：7月下旬-8月下旬
# 这是中国南方水稻受高温热害（SVPD负相关）最频繁的时段
ax.axvspan(16, 20, color='red', alpha=0.1, label='中稻关键期 (抽穗-扬花)')

# 晚稻 (抽穗-灌浆期)：9月中旬-10月中旬
# 此时期 SPEI 正相关可能对应干旱影响，SVPD 相关性通常减弱
ax.axvspan(22, 26, color='green', alpha=0.1, label='晚稻关键期 (抽穗-灌浆)')

# 设置 Y 轴
max_y = max(max(spei_pcts), max(svpd_pcts)) if (spei_pcts or svpd_pcts) else 50
ax.set_ylim(0, max_y + 10)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# --- 核心修改：图例移至左上角，并分两列以节省垂直空间 ---
ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, ncol=2)

plt.tight_layout()
save_path = os.path.join(output_dir, 'Rice_Types_Sensitivity_Comparison.png')
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"✅ 绘图成功！已区分早/中/晚稻阴影。文件保存至: {save_path}")