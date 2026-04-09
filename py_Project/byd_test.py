from Reservoir_Rulescheduling import *
import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\wangrui\Desktop\硕士毕业文件\毕业论文数据\水库设计洪水.xlsx'
sheet_name1 = '横山岭优化'
sheet_name2 = '龙门优化'
sheet_name3 = '王快优化'
sheet_name4 = '西大洋优化'
sheet_name5 = '区间入流'

# 使用 pandas 读取指定的 Excel 表格
df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
df3 = pd.read_excel(file_path, sheet_name=sheet_name3)
df4 = pd.read_excel(file_path, sheet_name=sheet_name4)
df5 = pd.read_excel(file_path, sheet_name=sheet_name5)

ll = 0  # 优化 0 1 2 规程 1 2 3
l1 = 0 # 0 20年 1 50年 2 100年
# 提取第一列数据
Q1 = df1.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q1 = np.round(Q1, 2)
Q2 = df2.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
Q3 = df3.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q3 = np.round(Q3, 2)
Q4 = df4.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q4 = np.round(Q4, 2)
Q5 = df5.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q5 = np.round(Q5, 2)

if l1 == 0:
    out_q1 = muskingum_non_linear(Q1, 14, 0.29, 3, Q1[0], 1)  # 马棚淀 横山岭
    out_q2 = muskingum_non_linear(Q2, 10, 0.35, 3, Q2[0], 1)  # 藻杂淀系统 龙门
    out_q3 = muskingum_non_linear(Q3, 14, 0.29, 3, Q3[0], 1)  # 马棚淀   王快
    out_q4 = muskingum_non_linear(Q4, 19, 0.37, 3, Q4[0], 1)  # 唐河系统 西大洋
elif l1 == 1:
    out_q1 = muskingum_non_linear(Q1, 10, 0.23, 3, Q1[0], 1)
    out_q2 = muskingum_non_linear(Q2, 8, 0.29, 3, Q2[0], 1)
    out_q3 = muskingum_non_linear(Q3, 10, 0.23, 3, Q3[0], 1)
    out_q4 = muskingum_non_linear(Q4, 15, 0.3, 3, Q4[0], 1)
elif l1 == 2:
    out_q1 = muskingum_non_linear(Q1, 8, 0.2, 3, Q1[0], 1)
    out_q2 = muskingum_non_linear(Q2, 6, 0.24, 3, Q2[0], 1)
    out_q3 = muskingum_non_linear(Q3, 8, 0.2, 3, Q3[0], 1)
    out_q4 = muskingum_non_linear(Q4, 10, 0.26, 3, Q4[0], 1)

q1_array = np.array(out_q1)
q2_array = np.array(out_q2)
q3_array = np.array(out_q3)
q4_array = np.array(out_q4)

q = q1_array + q2_array + q3_array + q4_array + Q5
z6_list, q6_list, qzw, qwa = byd(q)
print("白洋淀入淀洪水洪峰:", max(q))
print("白洋淀最高水位:", max(z6_list))
print("白洋淀水位:", z6_list)
print("白洋淀周围分洪:", qzw)
print("文安洼分洪:", qwa)
print("枣林庄泄流过程:", q6_list)

integers = list(range(1, len(q4_array) + 1))  # 生成1到Q长度的整数列表
x = integers
y1 = z6_list  # 第一条折线的纵坐标数据
y2 = q6_list  # 第二条折线的纵坐标数据
y3 = q  # 入淀流量
y4 = qzw

#
# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制第一条折线
ax1.plot(x, y2, 'r-', label='下泄流量')  # 'r-' 表示红色的实线
ax1.plot(x, y3, 'g-', label='入淀流量')  # 'g-' 表示绿色的实线
# ax1.plot(x, y4, 'k-', label='分洪流量')  # 'r-' 表示黑色的实线
ax1.set_xlabel('时间段(t=3h)')  # 设置x轴标签
ax1.set_ylabel(r'流量($\mathrm{m}^3$/s)', color='k')  # 设置y1轴标签
ax1.tick_params(axis='y', labelcolor='k')  # 设置y1轴标签的颜色

# 创建第二个纵坐标轴（与第一个共享x轴）
ax2 = ax1.twinx()
# 绘制第二条折线
ax2.plot(x, y1, 'b--', label='水位')  # 'b-' 表示蓝色的实线
ax2.set_ylabel('水位（m）', color='b')  # 设置y2轴标签
ax2.tick_params(axis='y', labelcolor='b')  # 设置y2轴标签的颜色
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.95))

# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()
