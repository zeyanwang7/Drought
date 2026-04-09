import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\水库优化调度结果.xlsx'
sheet_name1 = '王快优化'
sheet_name2 = '西大洋优化'
sheet_name3 = '龙门优化'
sheet_name4 = '横山岭优化'

# 使用 pandas 读取指定的 Excel 表格
df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
df3 = pd.read_excel(file_path, sheet_name=sheet_name3)
df4 = pd.read_excel(file_path, sheet_name=sheet_name4)

ll = 4  # 0 4 8
Q1 = df4.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q1 = np.round(Q1, 2)
Q2 = df4.iloc[:, ll+1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
Z1 = df4.iloc[:, ll+2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Z1 = np.round(Z1, 2)

integers = list(range(1, len(Q1) + 1))  # 生成1到Q长度的整数列表
x = integers
y1 = Z1  # 第一条折线的纵坐标数据
y2 = Q2   # 第二条折线的纵坐标数据
y3 = Q1 # 入库流量

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
ax1.set_ylabel(r'流量($\mathrm{m}^3$/s)', color='g')  # 设置y1轴标签
ax1.tick_params(axis='y', labelcolor='k')  # 设置y1轴标签的颜色

# 创建第二个纵坐标轴（与第一个共享x轴）
ax2 = ax1.twinx()
# 绘制第二条折线
ax2.plot(x, y1, 'b--', label='水位')  # 'b-' 表示蓝色的实线
ax2.set_ylabel('水位（m）', color='b')  # 设置y2轴标签
ax2.tick_params(axis='y', labelcolor='b')  # 设置y2轴标签的颜色
# fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95))
fig.legend(loc='upper right', bbox_to_anchor=(0.33, 0.95))
# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()


