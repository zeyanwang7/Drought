from Reservoir_Rulescheduling import *
import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd
import matplotlib as mpl

# 设置全局字体（Windows系统常用字体）
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']  # 优先使用微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = r'C:\Users\wangrui\Desktop\硕士毕业文件\毕业论文数据\水库设计洪水.xlsx'
sheet_name1 = '王快'
sheet_name2 = '横山岭'
sheet_name3 = '西大洋'
sheet_name4 = '龙门'

# 使用 pandas 读取指定的 Excel 表格
df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
df3 = pd.read_excel(file_path, sheet_name=sheet_name3)
df4 = pd.read_excel(file_path, sheet_name=sheet_name4)

ll = 1 # 0,1,2
# 提取数据
Q1 = df1.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q1 = np.round(Q1, 2)
z1_list, q1_list = wk_reservoir(Q1)
# 输出王快水库的最大值


Q2 = df2.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
z2_list, q2_list = hsl_reservoir(Q2)
# 输出横山岭水库的最大值


Q3 = df3.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q3 = np.round(Q3, 2)
z3_list, q3_list = xdy_reservoir(Q3)
# 输出西大洋水库的最大值


Q4 = df4.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q4 = np.round(Q4, 2)
z4_list, q4_list = lm_reservoir(Q2)
# 输出龙门水库的最大值


print("西大洋水库结果：")
print(f"z3_list 最大值: {max(z3_list):.2f}")
print(f"q3_list 最大值: {max(q3_list):.2f}")
print("-" * 30)
print("王快水库结果：")
print(f"z1_list 最大值: {max(z1_list):.2f}")
print(f"q1_list 最大值: {max(q1_list):.2f}")
print("-" * 30)
print("龙门水库结果：")
print(f"z4_list 最大值: {max(z4_list):.2f}")
print(f"q4_list 最大值: {max(q4_list):.2f}")
print("-" * 30)
print("横山岭水库结果：")
print(f"z2_list 最大值: {max(z2_list):.2f}")
print(f"q2_list 最大值: {max(q2_list):.2f}")
print("-" * 30)
# if ll == 0:
#     k1 = 14 # 王快 横山岭  马棚淀
#     x1 = 0.29
#     k2 = 19 # 西大洋  唐河
#     x2 = 0.37
#     k3 = 10 # 龙门  藻杂淀
#     x3 = 0.35
# elif ll== 1:
#     k1 = 10  # 王快 横山岭  马棚淀
#     x1 = 0.23
#     k2 = 15  # 西大洋  唐河
#     x2 = 0.3
#     k3 = 8  # 龙门  藻杂淀
#     x3 = 0.29
# elif ll == 2:
#     k1 = 8  # 王快 横山岭  马棚淀
#     x1 = 0.2
#     k2 = 10  # 西大洋  唐河
#     x2 = 0.26
#     k3 = 6  # 龙门  藻杂淀
#     x3 = 0.24
# out_q1 = muskingum_non_linear(q1_list, k1, x1, k1, q1_list[0], 1)
# out_q2 = muskingum_non_linear(q2_list, k1, x1, k1, q2_list[0], 1)
# out_q3 = muskingum_non_linear(q3_list, k2, x2, k2, q3_list[0], 1)
# out_q4 = muskingum_non_linear(q4_list, k3, x3, k3, q4_list[0], 1)
#
# integers = list(range(1, len(out_q2) + 1))  # 生成1到Q长度的整数列表
# x = integers
#
# # ==================== 修改后的代码：将所有数据保存到同一工作表 ====================
# # 创建数据字典，将所有数据放在一起
# data_to_save = {
#     '时间段(t=3h)': x,
#     '王快水库水位(m)': z1_list,
#     '王快水库原始下泄流量(m³/s)': Q1,
#     '王快水库演进后流量(m³/s)': out_q1,
#
#     '横山岭水库水位(m)': z2_list,
#     '横山岭水库原始下泄流量(m³/s)': Q2,
#     '横山岭水库演进后流量(m³/s)': out_q2,
#
#     '西大洋水库水位(m)': z3_list,
#     '西大洋水库原始下泄流量(m³/s)': Q3,
#     '西大洋水库演进后流量(m³/s)': out_q3,
#
#     '龙门水库水位(m)': z4_list,
#     '龙门水库原始下泄流量(m³/s)': Q4,
#     '龙门水库演进后流量(m³/s)': out_q4
# }
#
# # 创建DataFrame
# df_output = pd.DataFrame(data_to_save)
#
# # 指定输出文件路径
# output_file_path = r'C:\Users\wangrui\Desktop\硕士毕业文件\小论文数据.xlsx'
#
# # 保存到Excel文件
# df_output.to_excel(output_file_path, index=False, sheet_name='规程调度水库数据')
# print(f"所有水库数据已成功保存至: {output_file_path}")