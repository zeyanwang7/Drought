import numpy as np
from matplotlib import pyplot as plt

z = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
     153]  # 西大洋水库水位#
v = [2.404, 2.641, 2.895, 3.176, 3.463, 3.765, 4.081, 4.432, 4.795, 5.167, 5.564, 5.972, 6.412, 6.874, 7.353, 7.836,
     8.331, 8.859, 9.407, 9.965, 10.531, 11.139, 11.746, 12.417]  # 西大洋水库水位对应库容
q = [185, 188, 252, 463, 795, 1211, 1696, 2235, 2822, 3576, 4471, 5464, 6569, 7795, 9070, 10364, 11715, 13125, 14592,
     16113, 17687, 19311, 20986, 22707]  # 西大洋水库下泄流量能力
Q = [100, 127, 155, 183, 210, 237, 265, 269, 273, 276, 280, 360, 840, 740, 720, 720, 760, 860, 2060, 2900, 4100, 6100,
     4200, 5800, 7940, 6200, 4700, 3800, 2700, 2140, 1700, 1360, 1140, 980, 860, 750, 690, 630, 590, 540, 510,
     480, 460, 430, 410, 390, 380, 360, 340, 330, 320, 310, 300, 290, 290, 290, 270, 266, 260, 257, 255, 252, 249,
     246]  # 西大洋水库入库流量


def process_list(input_list):
    """计算相邻元素平均值并在开头添加0"""
    averaged_list = [(input_list[i] + input_list[i + 1]) / 2 for i in range(len(input_list) - 1)]
    averaged_list.insert(0, 0)
    return averaged_list


Qj = process_list(Q)
Qj = np.flip(Qj)


# 线性插值函数
def linear_interpolate(x, y, x_new):
    if x_new < x[0] or x_new > x[-1]:
        raise ValueError("Interpolation input is out of range!")
    return np.interp(x_new, x, y)


# 动态规划算法
def flood_control_dp(time_steps, z_min, z_max, z0):
    """
    动态规划实现水库防洪调度
    :param time_steps: 时间步长数
    :param z_min: 最低水位
    :param z_max: 最高水位
    :param z0: 初始水位
    :return: 最优水位路径和最优下泄流量路径
    """
    # 离散化水位范围
    z_values = np.linspace(z_min, z_max, len(Q))

    # 初始化动态规划表和路径表
    dp_table = np.full((time_steps, len(z_values)), np.inf)
    decision_table = np.zeros((time_steps, len(z_values)))

    # 初始条件
    dp_table[0, :] = 0  # 第一步的目标值设为0

    # 初始化水库水量
    initial_volume = linear_interpolate(z, v, z0)  # 初始水库水量
    current_volume = initial_volume  # 当前水库水量

    # 动态规划递推
    for t in range(1, time_steps):
        inflow = Qj[t]  # 当前时段入库流量
        for i, z_curr in enumerate(z_values):
            for j, z_prev in enumerate(z_values):
                # 库容变化计算
                v_curr = linear_interpolate(z, v, z_curr)
                v_prev = linear_interpolate(z, v, z_prev)

                # 计算下泄流量
                outflow = (v_curr - v_prev) * 1e8 / (3 * 3600) + inflow
                max_outflow = linear_interpolate(z, q, z_curr)

                # 检查约束条件
                if z_min <= z_curr <= z_max and 50 <= outflow <= max_outflow:
                    cost = dp_table[t - 1, j] + outflow ** 2  # 目标函数为下泄流量的平方和
                    if cost < dp_table[t, i]:
                        dp_table[t, i] = cost
                        decision_table[t, i] = j  # 记录前一时刻最优水位索引

        # 更新水库水量
        current_volume += (inflow - outflow) * 3600 * 3 / 1e8  # 每个时间步进1小时，入库流量为 m³/s，转换为 m³/h
        water_level = np.interp(current_volume, v, z)  # 根据水量计算水位

        # 更新动态规划中的水位信息
        # 这个水位在每个时间步都会被计算并用于优化
        z_values[i] = water_level

    # 反向追踪最优路径
    best_path = np.zeros(time_steps)
    best_index = np.argmin(dp_table[-1, :])  # 找到最终时段的最优水位索引
    best_path[-1] = z_values[best_index]

    for t in range(time_steps - 2, -1, -1):
        best_index = int(decision_table[t + 1, best_index])
        best_path[t] = z_values[best_index]

    # 计算最优下泄流量路径
    best_outflows = []
    for t in range(1, time_steps):
        v_curr = linear_interpolate(z, v, best_path[t])
        v_prev = linear_interpolate(z, v, best_path[t - 1])
        outflow = (v_curr - v_prev) * 1e8 / (3 * 3600) + Qj[t]
        best_outflows.append(outflow)

    return best_path, best_outflows


# 参数设置
time_steps = len(Q)  # 时间步长
z_min = 134.5  # 最低水位
z_max = 147.5  # 最高水位
z0 = 134.5  # 初始水位

# 运行动态规划
optimal_water_levels, optimal_outflows = flood_control_dp(time_steps, z_min, z_max, z0)

# 确保水位和下泄流量在时间顺序上对齐
optimal_water_levels = np.array(optimal_water_levels)
optimal_outflows = np.array(optimal_outflows)

# 计算最大下泄能力
max_discharge = np.array([linear_interpolate(z, q, wl) for wl in optimal_water_levels])
max_discharge = np.array(max_discharge)

optimal_outflows = np.flip(optimal_outflows)
optimal_water_levels = np.flip(optimal_water_levels)
max_discharge = np.flip(max_discharge)
Qj = np.flip(Qj)

# 结果可视化 - 水位变化
plt.figure(figsize=(12, 6))
plt.plot(range(time_steps), optimal_water_levels, label='Optimal Water Levels (m)', marker='o', color='blue')
plt.title("Optimal Water Levels Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Water Level (m)")
plt.legend()
plt.grid()
plt.show()

# 结果可视化 - 下泄流量变化
plt.figure(figsize=(12, 6))
plt.plot(range(1, time_steps), optimal_outflows, label='Optimal Outflows (m³/s)', marker='o', color='red')
plt.plot(range(1, time_steps), max_discharge[1:], label='Max Discharge Capacity (m³/s)', marker='x', color='blue',
         linestyle='--')
plt.plot(range(len(Qj)), Qj, marker='o', color='g', label='Inflow (Q)')
plt.title("Optimal Outflows (m³/s)")
plt.xlabel("Time Steps")
plt.ylabel("Optimal Outflows (m³/s)")
plt.legend()
plt.grid()
plt.show()


def muskingum_non_linear(inflow, K, x, dt, O0, n):
    """
    非线性马斯京根法计算流量演算。

    参数：
    inflow: list or numpy array，入流量序列。
    K: float，马斯京根储存参数。
    x: float，权重系数，范围为0到0.5。
    dt: float，时间步长。
    O0: float，初始出流量。
    n: float，非线性指数。

    返回：
    outflow: numpy array，出流量序列。
    """
    # 参数计算
    C0 = (dt - 2 * K * x) / (2 * K * (1 - x) + dt)
    C1 = (dt + 2 * K * x) / (2 * K * (1 - x) + dt)
    C2 = (2 * K * (1 - x) - dt) / (2 * K * (1 - x) + dt)

    # 初始化出流量序列
    outflow = [O0]

    # 逐时间步计算出流量
    for t in range(1, len(inflow)):
        # 非线性修正
        O_prev = outflow[-1]
        O_t = (C0 * inflow[t] + C1 * inflow[t - 1] + C2 * O_prev) ** n
        outflow.append(O_t)

    return np.array(outflow)


# 测试参数
inflow = optimal_outflows  # 入流量序列
K = 15  # 储存参数
x = 0.3  # 权重系数
dt = 3  # 时间步长
O0 = 200  # 初始出流量
n = 1  # 非线性指数

# 调用函数计算
outflow = muskingum_non_linear(inflow, K, x, dt, O0, n)

# 可视化结果
time = np.arange(len(inflow))
plt.plot(time, inflow, label="Inflow", marker='o')
plt.plot(time, outflow, label="Outflow", marker='x')
plt.xlabel("Time Steps")
plt.ylabel("Flow Rate")
plt.title("Muskingum Method")
plt.legend()
plt.grid()
plt.show()
