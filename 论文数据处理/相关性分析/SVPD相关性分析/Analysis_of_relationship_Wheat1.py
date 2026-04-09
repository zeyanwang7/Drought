import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
svpd_path = r'E:\data\计算结果\Provinces_Wheat_SVPD_Steps_2000_2022.csv'
yield_path = r'E:\data\计算结果\表格\二次项去趋势标准化产量(SYRS).xlsx'
output_dir = r'E:\data\计算结果\相关性分析_SVPD'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 加载与名称标准化 ---
print("正在加载数据...")
# 1. 加载 SVPD 表并标准化列名
df_svpd = pd.read_csv(svpd_path, index_col=0, parse_dates=True)
df_svpd.columns = [c.strip()[:2] if c not in ['year', 'period'] else c for c in df_svpd.columns]

# 2. 加载产量表并标准化省份索引
df_yield_raw = pd.read_excel(yield_path, sheet_name='小麦', index_col=0)
df_yield_raw.index = [str(i).strip()[:2] for i in df_yield_raw.index]
df_yield = df_yield_raw.transpose()
df_yield.index = df_yield.index.astype(int)

# --- 3. 确定省份基准 (以 SVPD 表为准) ---
# 获取 SVPD 表中除功能列外的所有省份
svpd_provinces = [c for c in df_svpd.columns if c not in ['year', 'period']]
print(f"SVPD 表中共有 {len(svpd_provinces)} 个省份。")

# --- 4. 循环计算 Pearson 相关性 ---
print("开始匹配产量数据并计算相关性...")
corr_results = []
missing_in_yield = []

for prov in svpd_provinces:
    # 检查该省份是否在产量表中有对应列
    if prov not in df_yield.columns:
        missing_in_yield.append(prov)
        continue

    # 预处理阶段标记 (1-46)
    temp_svpd = df_svpd.copy()
    temp_svpd['year'] = temp_svpd.index.year
    temp_svpd['period'] = ((temp_svpd.index.dayofyear - 1) // 8 + 1)
    temp_svpd.loc[temp_svpd['period'] > 46, 'period'] = 46

    for p in range(1, 47):
        # 提取 SVPD 序列
        svpd_sub = temp_svpd[temp_svpd['period'] == p][['year', prov]].set_index('year')
        # 提取产量序列
        yield_sub = df_yield[prov]

        # 合并对齐
        combined = pd.concat([svpd_sub, yield_sub], axis=1).dropna()
        combined.columns = ['SVPD', 'Yield']

        if len(combined) >= 15:
            r, p_val = stats.pearsonr(combined['SVPD'], combined['Yield'])
            corr_results.append({
                'Province': prov,
                'Period': p,
                'R': r,
                'P_Value': p_val,
                'N_Years': len(combined)
            })

# --- 5. 打印对齐报告 ---
if missing_in_yield:
    print(f"⚠️ 以下省份在 SVPD 表中存在，但在产量表中未发现: {missing_in_yield}")
    print("请确认这些省份是否为非主要产麦区，或在产量 Excel 中名称是否有误。")

# 保存结果
df_res = pd.DataFrame(corr_results)
res_save_path = os.path.join(output_dir, 'Wheat_SVPD_Yield_Correlation.csv')
df_res.to_csv(res_save_path, index=False, encoding='utf-8-sig')

# --- 6. 绘图展示 (仅针对成功匹配的省份) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

success_provinces = df_res['Province'].unique()
print(f"成功计算相关性的省份数量: {len(success_provinces)}")

for prov in success_provinces:
    subset = df_res[df_res['Province'] == prov].sort_values('Period')

    plt.figure(figsize=(12, 5))
    plt.plot(subset['Period'], subset['R'], color='#e6550d', lw=2, marker='o', markersize=3, label='Pearson R')

    # 标注显著性点
    sig = subset[subset['P_Value'] < 0.05]
    if not sig.empty:
        plt.scatter(sig['Period'], sig['R'], color='black', s=50, label='P < 0.05', zorder=5)

    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.fill_between(subset['Period'], 0, subset['R'], where=(subset['R'] < 0), color='#fee0d2', alpha=0.6)

    plt.title(f'{prov} 小麦产量对全年生育期 SVPD 的敏感性分析', fontsize=14, fontweight='bold')
    plt.xlabel('生育阶段 (Period 1-46)', fontsize=12)
    plt.ylabel('相关系数 R', fontsize=12)
    plt.xticks(np.linspace(1, 46, 12), [f'{m}月' for m in range(1, 13)])
    plt.grid(True, alpha=0.3, ls=':')
    plt.legend(loc='lower right')

    plot_subdir = os.path.join(output_dir, 'Province_Plots_SVPD')
    if not os.path.exists(plot_subdir): os.makedirs(plot_subdir)
    plt.savefig(os.path.join(plot_subdir, f'{prov}_SVPD敏感性曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"✨ 处理完成！结果表已保存至: {res_save_path}")