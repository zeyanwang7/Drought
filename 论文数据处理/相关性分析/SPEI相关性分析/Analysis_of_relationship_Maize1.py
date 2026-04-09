import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
spei_path = r'E:\data\计算结果\Provinces_Maize_SPEI_Steps.csv'
yield_path = r'E:\data\计算结果\表格\二次项去趋势标准化产量(SYRS).xlsx'
output_dir = r'E:\data\计算结果\相关性分析'
os.makedirs(output_dir, exist_ok=True)

# --- 2. 加载数据 ---
print("正在加载数据...")
# SPEI 表：Index 为日期，Columns 为省份
df_spei = pd.read_csv(spei_path, index_col=0, parse_dates=True)

# 产量表：Index 为省份，Columns 为年份 (B1-X1)
# 转置后：Index 为年份，Columns 为省份
df_yield = pd.read_excel(yield_path, sheet_name='玉米', index_col=0).transpose()
df_yield.index = df_yield.index.astype(int)  # 确保年份索引为整数

# --- 3. SPEI 阶段标记 (Period 1-46) ---
df_spei['year'] = df_spei.index.year
# 每一个 8 天为一个周期
df_spei['period'] = ((df_spei.index.dayofyear - 1) // 8 + 1)
df_spei.loc[df_spei['period'] > 46, 'period'] = 46

# --- 4. 循环计算 Pearson 相关性 ---
print("开始计算各省分阶段相关性...")
corr_results = []
# 直接取两表共有的列名（省份）
common_provinces = [p for p in df_spei.columns if p in df_yield.columns and p not in ['year', 'period']]

for prov in common_provinces:
    for p in range(1, 47):
        # 提取该省所有年份中第 p 阶段的 SPEI 值
        spei_sub = df_spei[df_spei['period'] == p][['year', prov]].set_index('year')
        # 提取该省产量序列
        yield_sub = df_yield[prov]

        # 按年份对齐合并
        combined = pd.concat([spei_sub, yield_sub], axis=1).dropna()

        # 计算相关系数
        if len(combined) >= 15:  # 确保至少有15年的样本量
            r, p_val = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
            corr_results.append({
                'Province': prov,
                'Period': p,
                'R': r,
                'P_Value': p_val
            })

# 保存结果
df_res = pd.DataFrame(corr_results)
df_res.to_csv(os.path.join(output_dir, 'Maize_SPEI_Yield_Correlation.csv'), index=False, encoding='utf-8-sig')

# --- 5. 绘图展示 (选取代表性省份) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for prov in common_provinces:
    subset = df_res[df_res['Province'] == prov].sort_values('Period')
    if subset.empty: continue

    plt.figure(figsize=(12, 5))
    plt.plot(subset['Period'], subset['R'], color='#0571b0', lw=2, label='Pearson R')

    # 显著性标注
    sig = subset[subset['P_Value'] < 0.05]
    plt.scatter(sig['Period'], sig['R'], color='#ca0020', s=50, label='P < 0.05', zorder=5)

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title(f'{prov}省 玉米产量对全年生育期 SPEI 的敏感性分析', fontsize=14)
    plt.xlabel('阶段 (一年内 46 个 8 天周期)', fontsize=12)
    plt.ylabel('相关系数 R', fontsize=12)

    # 设置 X 轴为月份刻度
    plt.xticks(np.linspace(1, 46, 12), [f'{m}月' for m in range(1, 13)])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(output_dir, f'{prov}_敏感性曲线.png'), dpi=300)
    plt.close()

print(f"✅ 计算完成！结果大表及各省敏感性图已保存至: {output_dir}")