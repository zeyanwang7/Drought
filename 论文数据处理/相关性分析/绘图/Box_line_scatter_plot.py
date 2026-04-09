import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 数据加载与整合 ---
# 假设您的文件路径如下，请根据实际情况微调
file_dict = {
    'SPEI (土壤水分干旱)': r'E:\data\计算结果\相关性分析\Wheat_SPEI_Yield_Correlation.csv',
    'SVPD (大气干热压力)': r'E:\data\计算结果\相关性分析_SVPD\Wheat_SVPD_Yield_Correlation.csv'
}

all_data = []
for name, path in file_dict.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Index_Type'] = name
        all_data.append(df)

df_all = pd.concat(all_data, axis=0)

# --- 2. 筛选 3-6 月关键生育窗口 ---
# 小麦产量对气象因子的响应主要集中在春季到初夏
target_periods = list(range(9, 25))
df_target = df_all[df_all['Period'].isin(target_periods)].copy()

# 将 Period 转换为月份标签
df_target['Month'] = df_target['Period'].apply(lambda x: f"{(x-1)//4 + 1}月")

# --- 3. 绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(13, 8), dpi=300)

# 定义颜色：蓝色系代表水分，红色系代表热力
my_pal = {"SPEI (土壤水分干旱)": "#3182bd", "SVPD (大气干热压力)": "#e6550d"}

# A. 绘制箱线图 (Boxplot) - 展示全国统计特征
# showfliers=False 隐藏原始离群点符号，因为我们要用后面的 stripplot 绘制所有真实点
sns.boxplot(data=df_target, x="Month", y="R", hue="Index_Type",
            palette=my_pal, width=0.6, showfliers=False,
            boxprops=dict(alpha=0.2, edgecolor='black'), # 箱体半透明
            whiskerprops=dict(color='gray'),
            medianprops=dict(color='black', lw=2))

# B. 绘制散点图 (Stripplot) - 展示所有省份真实样本
# dodge=True 让点跟随箱子分类对齐，jitter=True 增加随机抖动防止点完全重叠
sns.stripplot(data=df_target, x="Month", y="R", hue="Index_Type",
              palette=my_pal, dodge=True, jitter=True,
              size=5, alpha=0.6, linewidth=0.5, legend=False)

# --- 4. 图表修饰 ---
plt.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
plt.title('全国尺度下小麦产量对 SPEI 与 SVPD 响应强度的分布对比', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('相关系数 (Pearson R)', fontsize=12)
plt.xlabel('生育月份', fontsize=12)

# 设置 y 轴范围，通常相关系数在 -1 到 1 之间，农业研究多在 -0.8 到 0.6
plt.ylim(-0.8, 0.6)

# 添加网格线以便对齐
plt.grid(axis='y', linestyle=':', alpha=0.5)

# 优化图例
plt.legend(title="监测指数", loc='lower right', frameon=True, shadow=True)

# 保存图片
output_path = r'E:\data\计算结果\对比分析\National_Sample_Box_Scatter_Comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"✨ 全国大样本叠加图已生成：{output_path}")