import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
spei_path = r'E:\data\计算结果\Provinces_Wheat_SPEI_Steps.csv'
yield_path = r'E:\data\计算结果\表格\二次项去趋势标准化产量(SYRS).xlsx'
output_dir = r'E:\data\计算结果\相关性分析'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 加载与名称标准化 ---
print("正在加载数据...")
# SPEI 表：列名可能包含“省/自治区”，将其统一为前2个字
df_spei = pd.read_csv(spei_path, index_col=0, parse_dates=True)
# 修正列名：去除空格，保留前两个字，同时保留功能列
df_spei.columns = [c.strip()[:2] if c not in ['year', 'period'] else c for c in df_spei.columns]

# 产量表：加载并标准化省份索引
df_yield_raw = pd.read_excel(yield_path, sheet_name='小麦', index_col=0)
# 统一索引名为前2个字
df_yield_raw.index = [str(i).strip()[:2] for i in df_yield_raw.index]
# 转置后：Index 为年份，Columns 为省份
df_yield = df_yield_raw.transpose()
df_yield.index = df_yield.index.astype(int)  # 确保年份索引为整数

# --- 3. SPEI 阶段标记 (Period 1-46) ---
df_spei['year'] = df_spei.index.year
# 每一个 8 天为一个周期
df_spei['period'] = ((df_spei.index.dayofyear - 1) // 8 + 1)
df_spei.loc[df_spei['period'] > 46, 'period'] = 46

# --- 4. 寻找共有省份并计算相关性 ---
print("正在对齐省份列表...")
common_provinces = [p for p in df_spei.columns if p in df_yield.columns and p not in ['year', 'period']]

print(f"成功识别共有省份 {len(common_provinces)} 个: {common_provinces}")
if '青海' not in common_provinces:
    print("⚠️ 警告：青海仍未匹配成功！")
    print(f"SPEI 表省份示例: {list(df_spei.columns[:10])}")
    print(f"产量表省份示例: {list(df_yield.columns[:10])}")

print("开始计算 Pearson 相关性...")
corr_results = []

for prov in common_provinces:
    valid_periods = 0
    for p in range(1, 47):
        # 提取该省在特定阶段的 SPEI 序列
        spei_sub = df_spei[df_spei['period'] == p][['year', prov]].set_index('year')
        # 提取该省产量序列
        yield_sub = df_yield[prov]

        # 按年份对齐合并 (内连接)
        combined = pd.concat([spei_sub, yield_sub], axis=1).dropna()
        combined.columns = ['SPEI', 'Yield']

        # 确保样本量，通常 15 年以上具有统计学意义
        if len(combined) >= 15:
            r, p_val = stats.pearsonr(combined['SPEI'], combined['Yield'])
            corr_results.append({
                'Province': prov,
                'Period': p,
                'R': r,
                'P_Value': p_val,
                'N_Years': len(combined)
            })
            valid_periods += 1

    if prov == '青海':
        print(f"✅ 青海省处理完成，有效样本年份: {len(combined)}，成功计算 {valid_periods} 个阶段。")

# 保存结果
df_res = pd.DataFrame(corr_results)
res_save_path = os.path.join(output_dir, 'Wheat_SPEI_Yield_Correlation.csv')
df_res.to_csv(res_save_path, index=False, encoding='utf-8-sig')

# --- 5. 绘图展示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for prov in common_provinces:
    subset = df_res[df_res['Province'] == prov].sort_values('Period')
    if subset.empty:
        print(f"跳过 {prov}: 无有效计算结果。")
        continue

    plt.figure(figsize=(12, 5))
    # 绘制 R 值曲线
    plt.plot(subset['Period'], subset['R'], color='#0571b0', lw=2, marker='o', markersize=3, label='Pearson R')

    # 显著性点标注 (P < 0.05)
    sig = subset[subset['P_Value'] < 0.05]
    if not sig.empty:
        plt.scatter(sig['Period'], sig['R'], color='#ca0020', s=60, label='P < 0.05', zorder=5, edgecolors='white')

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title(f'{prov} 小麦产量对全年生育期 SPEI 的敏感性分析', fontsize=14, fontweight='bold')
    plt.xlabel('生育阶段 (Period 1-46)', fontsize=12)
    plt.ylabel('相关系数 R', fontsize=12)

    # X轴刻度：显示月份对应关系
    month_ticks = np.linspace(1, 46, 12)
    plt.xticks(month_ticks, [f'{int(m)}月' for m in range(1, 13)])

    plt.grid(True, alpha=0.3, ls=':')
    plt.legend(loc='upper right')

    # 子目录用于存放大量图片
    plot_dir = os.path.join(output_dir, 'Province_Plots')
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    plt.savefig(os.path.join(plot_dir, f'{prov}_敏感性曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"✨ 任务全部完成！\n1. 结果表格: {res_save_path}\n2. 趋势图目录: {plot_dir}")