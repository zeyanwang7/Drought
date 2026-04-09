# 各水库下泄洪水演进到白洋淀过程
# 仅作为绘图使用
from Reservoir_Rulescheduling import *
import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd

import matplotlib as mpl

# 设置全局字体（Windows系统常用字体）
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']  # 优先使用微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 之后的所有绘图都会自动使用中文字体

# 或者指定其他中文字体（按优先级尝试）
# font_list = ['Microsoft YaHei', 'SimSun', 'SimHei', 'KaiTi', 'FangSong']
# mpl.rcParams['font.sans-serif'] = font_list
# 读取Excel文件
file_path = r'C:\Users\wangrui\Desktop\硕士毕业文件\毕业论文数据\水库设计洪水.xlsx'
# sheet_name1 = '龙门优化'
# sheet_name2 = '龙门优化'
sheet_name3 = '横山岭优化'
# sheet_name4 = '西大洋优化'
# sheet_name5 = '区间入流'

# 使用 pandas 读取指定的 Excel 表格
# df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
df2 = pd.read_excel(file_path, sheet_name=sheet_name3)

ll = 2 # 0,1,2
# 提取数据
# Q1 = df1.iloc[:, 1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
# Q1 = np.round(Q1, 2)

Q2 = df2.iloc[:, 2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
z2_list, q2_list = wk_reservoir(Q2)

k = 8
x1 = 0.2
# out_q1 = muskingum_non_linear(Q1, k, x1, k, Q1[0], 1)
out_q2 = muskingum_non_linear(q2_list, k, x1, k, q2_list[0], 1)

integers = list(range(1, len(out_q2) + 1))  # 生成1到Q长度的整数列表
x = integers
# y1 = Q1  # 第一条折线的纵坐标数据
# y2 = out_q1  # 第二条折线的纵坐标数据
y3 = q2_list
y4 = out_q2

# 设置全局字体（加粗效果）
# 设置全局字体（修正版）
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],  # 中文用黑体
    'axes.unicode_minus': False,  # 解决负号显示问题
    'font.weight': 'bold',  # 全局字体加粗
    'axes.titleweight': 'bold',  # 标题加粗
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'xtick.labelsize': 14,  # x轴刻度字号
    'ytick.labelsize': 14,  # y轴刻度字号
    # 注意：legend.fontweight 不是有效参数，改为在legend()中单独设置
})

# 创建图形和坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制两条折线（带不同样式）
line1 = ax1.plot(x, y3, 'g-+', markersize=8, linewidth=2, label='规程调度原始下泄流量')
line2 = ax1.plot(x, y4, 'r-', linewidth=2, label='规程调度演进后流量')

# 设置坐标轴标签
ax1.set_xlabel('时间段 (t=3h)', fontsize=14)
ax1.set_ylabel(r'流量 ($\mathrm{m}^3$/s)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='k')

# 加粗坐标轴线（2.0磅）
ax1.spines['bottom'].set_linewidth(2.0)  # 下边框
ax1.spines['left'].set_linewidth(2.0)  # 左边框
ax1.spines['top'].set_linewidth(2.0)  # 上边框
ax1.spines['right'].set_linewidth(2.0)  # 右边框

# 添加图例
legend = ax1.legend(
    prop={'weight': 'bold', 'size': 14},  # 图例文本加粗
    framealpha=1,
    shadow=False,
    edgecolor='black',
    bbox_to_anchor=(0.65, 0.99),
    borderaxespad=0.5
)  # <-- 这里必须有闭合括号

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()
