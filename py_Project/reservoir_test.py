import numpy as np
from matplotlib import rcParams, pyplot as plt
from Reservoir_Rulescheduling import *
import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\wangrui\Desktop\硕士毕业文件\毕业论文数据\水库设计洪水.xlsx'
sheet_name1 = '横山岭'
sheet_name2 = '龙门'
sheet_name3 = '王快'
sheet_name4 = '西大洋'
sheet_name5 = '区间入流'

# 使用 pandas 读取指定的 Excel 表格
# df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
# df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
# df3 = pd.read_excel(file_path, sheet_name=sheet_name3)
df4 = pd.read_excel(file_path, sheet_name=sheet_name4)
# df5 = pd.read_excel(file_path, sheet_name=sheet_name5)

ll = 2
# 提取第一列数据
# Q1 = df1.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
# Q1 = np.round(Q1, 2)
# Q2 = df2.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
# Q2 = np.round(Q2, 2)
# Q3 = df3.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
# Q3 = np.round(Q3, 2)
Q4 = df4.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q4 = np.round(Q4, 2)

# z1_list, q1_list = hsl_reservoir(Q1)
# z2_list, q2_list = lm_reservoir(Q2)
# z3_list, q3_list = wk_reservoir(Q3)
z4_list, q4_list = xdy_reservoir(Q4)

integers = list(range(1, len(q4_list) + 1))  # 生成1到Q长度的整数列表
x = integers
y1 = z4_list  # 第一条折线的纵坐标数据
y2 = q4_list  # 第二条折线的纵坐标数据
y3 = Q4  # 入库流量

#
# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制第一条折线
ax1.plot(x, y3, 'g-+', label='入库流量')  # 'g-' 表示绿色的实线
ax1.plot(x, y2, 'r-', label='下泄流量')  # 'r-' 表示红色的实线
ax1.set_xlabel('时间段(t=3h)')  # 设置x轴标签
ax1.set_ylabel(r'流量($\mathrm{m}^3$/s)', color='k')  # 设置y1轴标签
ax1.tick_params(axis='y', labelcolor='k')  # 设置y1轴标签的颜色

# 创建第二个纵坐标轴（与第一个共享x轴）
ax2 = ax1.twinx()
# 绘制第二条折线
ax2.plot(x, y1, 'b--', label='水位')  # 'b-' 表示蓝色的实线
ax2.set_ylabel('水位（m）', color='b')  # 设置y2轴标签
ax2.tick_params(axis='y', labelcolor='b')  # 设置y2轴标签的颜色
# fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95))
fig.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9))
# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()