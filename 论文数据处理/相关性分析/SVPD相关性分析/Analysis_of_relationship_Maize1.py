import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# --- 1. 路径设置 ---
# 使用之前生成的 SVPD 省份均值表
svpd_path = r'E:\data\计算结果\Provinces_Maize_SVPD_Steps_2000_2022.csv'
# 产量表路径
yield_path = r'E:\data\计算结果\表格\二次项去趋势标准化产量(SYRS).xlsx'
output_dir = r'E:\data\计算结果\相关性分析_SVPD'
os.makedirs(output_dir, exist_ok=True)

# --- 2. 加载数据 ---
print("正在加载数据...")
# SVPD 表：Index 为日期，Columns 为省份
df_svpd = pd.read_csv(svpd_path, index_col=0, parse_dates=True)

# 产量表：加载“水稻”页签
# 转置后：Index 为年份，Columns 为省份
try:
    df_yield = pd.read_excel(yield_path, sheet_name='玉米', index_col=0).transpose()
except Exception as e:
    print(f"读取产量表失败，请检查 Sheet 名称是否为 '玉米': {e}")
    # 如果读取失败，尝试读取第一个 sheet 或根据实际情况修改
    df_yield = pd.read_excel(yield_path, index_col=0).transpose()

df_yield.index = df_yield.index.astype(int)  # 确保年份索引为整数

# --- 3. 阶段标记 (Period 1-46) ---
df_svpd['year'] = df_svpd.index.year
# 每一个 8 天为一个周期 (1-46)
df_svpd['period'] = ((df_svpd.index.dayofyear - 1) // 8 + 1)
df_svpd.loc[df_svpd['period'] > 46, 'period'] = 46

# --- 4. 循环计算 Pearson 相关性 ---
print("开始计算各省分阶段相关性...")
corr_results = []
# 取共有省份，排除辅助列
common_provinces = [p for p in df_svpd.columns if p in df_yield.columns and p not in ['year', 'period']]

for prov in common_provinces:
    for p in range(1, 47):
        # 提取该省所有年份中第 p 阶段的 SVPD 值
        svpd_sub = df_svpd[df_svpd['period'] == p][['year', prov]].set_index('year')
        # 提取该省产量序列
        yield_sub = df_yield[prov]

        # 对齐年份合并
        combined = pd.concat([svpd_sub, yield_sub], axis=1).dropna()

        # 计算相关系数
        if len(combined) >= 15:  # 确保至少有 15 年的共有样本
            # combined.iloc[:, 0] 是 SVPD, combined.iloc[:, 1] 是产量
            r, p_val = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
            corr_results.append({
                'Province': prov,
                'Period': p,
                'R': r,
                'P_Value': p_val
            })

# 保存结果大表
df_res = pd.DataFrame(corr_results)
df_res.to_csv(os.path.join(output_dir, 'Maize_SVPD_Yield_Correlation.csv'), index=False, encoding='utf-8-sig')

# --- 5. 绘图展示 (生成各省敏感性曲线) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for prov in common_provinces:
    subset = df_res[df_res['Province'] == prov].sort_values('Period')
    if subset.empty: continue

    plt.figure(figsize=(12, 5))
    # 使用深红色表示干热压力
    plt.plot(subset['Period'], subset['R'], color='#d73027', lw=2, label='Pearson R (SVPD vs Yield)')

    # 显著性标注 (P < 0.05)
    sig = subset[subset['P_Value'] < 0.05]
    plt.scatter(sig['Period'], sig['R'], color='black', s=40, label='P < 0.05', zorder=5)

    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.title(f'{prov}省 玉米产量对全年生育期 SVPD 的敏感性分析', fontsize=14)
    plt.xlabel('阶段 (一年内 46 个 8 天周期)', fontsize=12)
    plt.ylabel('相关系数 R', fontsize=12)

    # 填充负相关区域（通常 SVPD 高会导致减产，所以 R<0 是我们关注的受损期）
    plt.fill_between(subset['Period'], 0, subset['R'], where=(subset['R'] < 0), color='#f46d43', alpha=0.2)

    # 设置 X 轴刻度对应月份
    plt.xticks(np.linspace(1, 46, 12), [f'{m}月' for m in range(1, 13)])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(output_dir, f'{prov}_SVPD敏感性曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ 计算完成！")
print(f"📊 相关性汇总表: {os.path.join(output_dir, 'Maize_SVPD_Yield_Correlation.csv')}")
print(f"📈 敏感性曲线图已保存至: {output_dir}")