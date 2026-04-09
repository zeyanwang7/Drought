import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\水库设计洪水 - 副本.xlsx'
sheet_name1 = '设计洪水20年'
sheet_name2 = '设计洪水50年'
sheet_name3 = '设计洪水100年'

# 使用 pandas 读取指定的 Excel 表格
df1 = pd.read_excel(file_path, sheet_name=sheet_name3)


ll = 0
# 提取第一列数据
Q1 = df1.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ，`tolist()` 将其转换为列表
Q1 = np.round(Q1, 2)
Q2 = df1.iloc[:, ll+1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
Q3 = df1.iloc[:, ll+2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q3 = np.round(Q3, 2)
Q4 = df1.iloc[:, ll+3].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q4 = np.round(Q4, 2)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建 x 轴数据（假设 x 轴为时间步长或索引）
x = range(len(Q1))  # x 轴数据，与列表长度一致

# 绘制折线图
plt.figure(figsize=(8, 5))  # 设置画布大小
plt.plot(x, Q1, label='王快水库', marker='o', linestyle='-', color='blue')  # 第一条折线
plt.plot(x, Q2, label='横山岭水库', marker='s', linestyle='--', color='green')  # 第二条折线
plt.plot(x, Q3, label='西大洋水库', marker='^', linestyle='-.', color='red')  # 第三条折线
plt.plot(x, Q4, label='龙门水库', marker='*', linestyle='-.', color='black')  # 第四条折线

# 添加标题和标签
plt.xlabel('时间段(t=3h)', fontsize=14)  # x 轴标签
plt.ylabel(r'流量($\mathrm{m}^3$/s)', fontsize=14)  # y 轴标签

# 添加图例
plt.legend(fontsize=12)

# 显示图表
plt.show()