from Reservoir_Rulescheduling import *
import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\水库设计洪水.xlsx'
sheet_name11 = '横山岭'
sheet_name21 = '龙门'
sheet_name31 = '王快'
sheet_name41 = '西大洋'
sheet_name51 = '区间入流'
sheet_name12 = '横山岭优化'
sheet_name22 = '龙门优化'
sheet_name32 = '王快优化'
sheet_name42 = '西大洋优化'
sheet_name52 = '区间优化'

# 使用 pandas 读取指定的 Excel 表格
df11 = pd.read_excel(file_path, sheet_name=sheet_name11)
df21 = pd.read_excel(file_path, sheet_name=sheet_name21)
df31 = pd.read_excel(file_path, sheet_name=sheet_name31)
df41 = pd.read_excel(file_path, sheet_name=sheet_name41)
df51 = pd.read_excel(file_path, sheet_name=sheet_name51)

df12 = pd.read_excel(file_path, sheet_name=sheet_name12)
df22 = pd.read_excel(file_path, sheet_name=sheet_name22)
df32 = pd.read_excel(file_path, sheet_name=sheet_name32)
df42 = pd.read_excel(file_path, sheet_name=sheet_name42)
df52 = pd.read_excel(file_path, sheet_name=sheet_name52)

ll1 = 3  # 1 2 3
ll2 = ll1 - 1  # 0 1 2
# 提取规程调度数据
Q11 = df11.iloc[:, ll1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q11 = np.round(Q11, 2)
Q21 = df21.iloc[:, ll1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q21 = np.round(Q21, 2)
Q31 = df31.iloc[:, ll1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q31 = np.round(Q31, 2)
Q41 = df41.iloc[:, ll1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q41 = np.round(Q41, 2)
Q51 = df51.iloc[:, ll1].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q51 = np.round(Q51, 2)

# 提取优化调度数据
Q12 = df12.iloc[:, ll2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q12 = np.round(Q12, 2)
Q22 = df22.iloc[:, ll2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q22 = np.round(Q22, 2)
Q32 = df32.iloc[:, ll2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q32 = np.round(Q32, 2)
Q42 = df42.iloc[:, ll2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q42 = np.round(Q42, 2)
Q52 = df52.iloc[:, ll2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q52 = np.round(Q52, 2)

out_q11 = muskingum_non_linear(Q11, 14, 0.29, 3, Q11[0], 1)  # 马棚淀 横山岭
out_q21 = muskingum_non_linear(Q21, 10, 0.35, 3, Q21[0], 1)  # 藻杂淀系统 龙门
out_q31 = muskingum_non_linear(Q31, 14, 0.29, 3, Q31[0], 1)  # 马棚淀   王快
out_q41 = muskingum_non_linear(Q41, 19, 0.37, 3, Q41[0], 1)  # 唐河系统 西大洋

out_q12 = muskingum_non_linear(Q12, 10, 0.23, 3, Q12[0], 1)
out_q22 = muskingum_non_linear(Q22, 8, 0.29, 3, Q22[0], 1)
out_q32 = muskingum_non_linear(Q32, 10, 0.23, 3, Q32[0], 1)
out_q42 = muskingum_non_linear(Q42, 15, 0.3, 3, Q42[0], 1)

q11_array = np.array(out_q11)
q21_array = np.array(out_q21)
q31_array = np.array(out_q31)
q41_array = np.array(out_q41)

q12_array = np.array(out_q12)
q22_array = np.array(out_q22)
q32_array = np.array(out_q32)
q42_array = np.array(out_q42)

q1 = q11_array + q21_array + q31_array + q41_array + Q51
q2 = q12_array + q22_array + q32_array + q42_array + Q52

z61_list, q61_list, qzw1, qwa1 = byd(q1)
z62_list, q62_list, qzw2, qwa2 = byd(q2)


def fl_js(z1):
    z = [5, 6, 7, 8, 8.5, 9, 10.5, 11]
    q11 = [42, 90, 124, 224, 325, 460, 630, 653]
    q21 = [0, 110, 370, 969, 1445, 1840, 2540, 2631]
    q31 = [0, 0, 0, 67, 220, 400, 1090, 1130]
    q1_list = []
    q2_list = []
    q3_list = []
    for zt in z1:
        q1_out = np.interp(zt, z, q11)
        q2_out = np.interp(zt, z, q21)
        q3_out = np.interp(zt, z, q31)
        q1_list.append(q1_out)
        q2_list.append(q2_out)
        q3_list.append(q3_out)
    return q1_list, q2_list, q3_list


q11_out, q21_out, q31_out = fl_js(z62_list)

print(q11_out)
print(q21_out)
print(q31_out)


integers = list(range(1, len(z61_list) + 1))  # 生成1到Q长度的整数列表
x = integers
y1 = q11_out  # 第一条折线的纵坐标数据
y2 = q21_out  # 第二条折线的纵坐标数据
y3 = q31_out

#
# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制第一条折线
ax1.plot(x, y1, 'r-', label='四孔闸泄洪流量')  # 'r-' 表示红色的实线
ax1.plot(x, y2, 'g-+', label='二十五孔闸泄洪流量')  # 'g-' 表示绿色的实线
ax1.plot(x, y3, 'b--', label='溢流堰泄洪流量')  # 'g-' 表示绿色的实线

# ax1.plot(x, y4, 'k-', label='分洪流量')  # 'r-' 表示黑色的实线
ax1.set_xlabel('时间段(t=3h)')  # 设置x轴标签
ax1.set_ylabel(r'流量($\mathrm{m}^3$/s)', color='k')  # 设置y1轴标签
ax1.tick_params(axis='y', labelcolor='k')  # 设置y1轴标签的颜色

fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.75))

# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()
