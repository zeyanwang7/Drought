import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\蓄滞洪区参数.xlsx'
sheet_name1 = '白洋淀'
sheet_name2 = '贾口洼'
sheet_name3 = '文安洼'
sheet_name4 = '东淀'



# 使用 pandas 读取指定的 Excel 表格
#df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
#df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
#df3 = pd.read_excel(file_path, sheet_name=sheet_name3)
df1 = pd.read_excel(file_path, sheet_name=sheet_name4)



# 提取第一列数据
z1 = df1.iloc[:, 0].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
z1 = np.round(z1, 1)

v1 = df1.iloc[:, 1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
v1 = np.round(v1, 3)

# q1 = df1.iloc[:, 2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
# q1 = np.round(q1, 0)
# 生成数据
print(z1)
print(v1)
# print(q1)
# x = z1  # x轴数据
# y = v1  # y轴数据
# #z = q1  # z轴数据（根据x和y生成z）
#
# # 绘制折线图
# plt.plot(x, y, linestyle='-', color='b')
#
# # 设置 Matplotlib 使用中文字体
# rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
# rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# # 设置坐标轴标签
# plt.xlabel('水位（m）', fontsize=12)
# plt.ylabel(r'库容(亿$\mathrm{m}^3$)', fontsize=12)
# #plt.ylabel(r'下泄流量($\mathrm{m}^3$/s)', fontsize=12)
# # 设置标题
#
# # 显示图形
# plt.show()
