import numpy as np
from matplotlib import rcParams

z = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
     153]  # 西大洋水库水位#
v = [2.404, 2.641, 2.895, 3.176, 3.463, 3.765, 4.081, 4.432, 4.795, 5.167, 5.564, 5.972, 6.412, 6.874, 7.353, 7.836,
     8.331, 8.859, 9.407, 9.965, 10.531, 11.139, 11.746, 12.417]  # 西大洋水库水位对应库容
q = [185, 188, 252, 463, 795, 1211, 1696, 2235, 2822, 3576, 4471, 5464, 6569, 7795, 9070, 10364, 11715, 13125, 14592,
     16113, 17687, 19311, 20986, 22707]  # 西大洋水库下泄流量能力
Q = [988.00, 2780.00, 3690.00, 4774.00, 5058.00, 4830.00, 5400.00, 4520.00, 3820.00, 3170.00, 2460.00, 1845.00, 1330.00,
     975.00, 820.00, 760.00, 735.00, 698.00, 625.00, 550.00, 487.00, 459.00, 430.00, 403.00, 396.00, 368.00, 322.00,
     303.00, 286.00, 280.00, 274.00, 266.00, 222.00, 210.00, 197.00]


def linear_interpolate(x, y, x_new):  # 线性插值
    return np.interp(x_new, x, y)


qx1 = linear_interpolate(z, q, 134.5)  # 水库起调水位下泄能力
vx1 = linear_interpolate(z, v, 134.5)  # 水库起调水位库容
print(qx1)
print(vx1)
vx = [vx1]
qx = [Q[0]]

for i in range(len(Q) - 1):
    sate = False
    Q1 = Q[i]
    Q2 = Q[i + 1]
    q1 = qx[-1]
    qt = qx[-1]
    while not sate:
        v2 = (Q1 + Q2) * 3 * 60 * 60 / 2 - (q1 + qt) * 3 * 60 * 60 / 2 + vx[-1] * 100000000
        v2 = v2 / 100000000
        q2 = linear_interpolate(v, q, v2)
        if abs(q2 - qt) < 1:
            sate = True
            qx.append(round(qt, 2))
            vx.append(round(v2, 2))
        else:
            qt = q2
print("qx", qx)
print("vx", vx)

print(len(qx))

integers = list(range(1, 36))  # 生成1到100的整数列表
print(integers)
x = integers
y1 = qx
y2 = Q
# 导入必需的库
import matplotlib.pyplot as plt

# 数据


# 创建一个图形和轴
plt.figure(figsize=(10, 6))  # 设置图形的大小

# 绘制折线图
plt.plot(x, y1, marker='o', linestyle='-', color='b', label='下泄流量')  # 下泄流量过程
plt.plot(x, y2, marker='s', linestyle='--', color='g', label='入库流量')  # 入库流量过程

# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# 添加标题和标签
plt.title('下泄流量过程')  # 图表标题
plt.xlabel('时间段')             # X 轴标签
plt.ylabel('流量（m3/s)')          # Y 轴标签


# 显示网格
plt.grid(True)

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()


