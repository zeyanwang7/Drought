import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

z = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
     153]  # 西大洋水库水位#
v = [2.404, 2.641, 2.895, 3.176, 3.463, 3.765, 4.081, 4.432, 4.795, 5.167, 5.564, 5.972, 6.412, 6.874, 7.353, 7.836,
     8.331, 8.859, 9.407, 9.965, 10.531, 11.139, 11.746, 12.417]  # 西大洋水库水位对应库容
q = [185, 188, 252, 463, 795, 1211, 1696, 2235, 2822, 3576, 4471, 5464, 6569, 7795, 9070, 10364, 11715, 13125, 14592,
     16113, 17687, 19311, 20986, 22707]  # 西大洋水库下泄流量能力
Q = [237.00, 354.00, 432.00, 166.00, 580.00, 610.00, 672.00, 665.00, 870.00, 988.00, 2780.00, 3690.00, 4774.00, 5058.00,
     4830.00, 5400.00, 4520.00, 3820.00, 3170.00, 2460.00, 1845.00, 1330.00, 975.00, 820.00, 760.00, 735.00, 698.00,
     625.00, 550.00, 487.00, 459.00, 430.00, 403.00, 396.00, 368.00, 322.00, 303.00, 286.00, 280.00, 274.00, 266.00,
     222.00, 210.00, 197.00]  # 西大洋水库入库流量


def linear_interpolate(x, y, x_new):  # 线性插值
    return np.interp(x_new, x, y)


print(len(Q))
zs = 133.35
qs = Q[0]

vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
v_list = [vs]
z_list = [zs]
q_list = [qs]
print(vs)

for i in range(len(Q) - 1):
    if 133.35 <= z_list[-1] < 139.43:
        qt = 300
    elif 139.43 <= z_list[-1] < 141.56:
        qt = 1000
    elif 141.56 <= z_list[-1] < 143.77:
        qt = 5460
    elif 143.77 <= z_list[-1] < 151.09:
        qt = Q[i]
    else:
        print("计算出错")
    vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
    v2 = vt / 100000000
    zt = linear_interpolate(v, z, v2)
    q_list.append(round(qt, 2))
    v_list.append(round(v2, 2))
    z_list.append(round(zt, 2))

print(z_list, len(z_list))
print(q_list, len(q_list))
print(v_list, len(v_list))


# integers = list(range(1, 45))  # 生成1到100的整数列表
# print(integers)
# x = integers
# y1 = q_list
# y2 = Q
# # 绘制折线图
# plt.plot(x, y1, marker='o', linestyle='-', color='b', label='下泄流量')  # 下泄流量过程
# plt.plot(x, y2, marker='s', linestyle='--', color='g', label='入库流量')  # 入库流量过程
#
# # 设置 Matplotlib 使用中文字体
# rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
# rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
#
# # 添加标题和标签
# plt.title('入流与泄流过程')  # 图表标题
# plt.xlabel('时间段')             # X 轴标签
# plt.ylabel('流量（m3/s)')          # Y 轴标签
#
#
# # 显示网格
# plt.grid(True)
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.tight_layout()  # 自动调整布局，避免标签重叠
# plt.show()

integers = list(range(1, len(Q)+1))  # 生成1到100的整数列表

x = integers
y1 = z_list  # 第一条折线的纵坐标数据
y2 = q_list  # 第二条折线的纵坐标数据
y3 = Q  # 入库流量

# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制第一条折线
ax1.plot(x, y1, 'g-', label='水位')  # 'g-' 表示绿色的实线
ax1.set_xlabel('时间段')  # 设置x轴标签
ax1.set_ylabel('水位（m）', color='g')  # 设置y1轴标签
ax1.tick_params(axis='y', labelcolor='g')  # 设置y1轴标签的颜色

# 创建第二个纵坐标轴（与第一个共享x轴）
ax2 = ax1.twinx()
# 绘制第二条折线
ax2.plot(x, y2, 'b-', label='下泄流量')  # 'b-' 表示蓝色的实线
ax2.plot(x, Q, 'r-', label='入库流量')  # 'b-' 表示蓝色的实线
ax2.set_ylabel('下泄流量', color='b')  # 设置y2轴标签
ax2.tick_params(axis='y', labelcolor='b')  # 设置y2轴标签的颜色


# 添加标题
plt.title('水位下泄流量变化过程')


# 添加图例
# 添加图例
ax1.legend(loc='upper left')  # 第一条折线图例在左上角
ax2.legend(loc='upper right')  # 第二条折线图例在右上角

# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()

