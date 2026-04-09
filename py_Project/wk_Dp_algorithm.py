import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 基本参数设置
z = [191,192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
     215]  # 水位
v = [3.733,4.03, 4.329, 4.644, 4.993, 5.301, 5.673, 6.049, 6.434, 6.834, 7.241, 7.66, 8.097, 8.575, 9.057, 9.549, 10.05,
     10.57, 11.136, 11.708, 12.288, 12.879, 13.484, 14.134, 14.789]  # 库容 (亿立方米)
q = [235,238, 451, 860, 1401, 2047, 2785, 3600, 4487, 5361, 6221, 7115, 8094, 9112, 10172, 11270, 12405, 13576, 14782,
     16020, 17294, 18599, 19936, 21303, 22701]  # 下泄流量能力 (立方米/秒)
# Q = [43, 61, 365, 232, 279, 640, 602, 408, 519, 927, 1065, 1159, 1863, 2621, 2321, 1566, 1713, 1391, 2686, 4771,
# 6104, 7513, 9600, 6794, 6714, 4527, 3885, 3494, 2787, 2293, 2051, 1503, 1480, 1400, 1229, 1080, 800, 722, 752, 900,
# 910, 725, 576, 520, 580, 600, 469, 480, 430, 380, 330, 290, 260, 262, 264, 267, 269, 270, 273, 276, 278, 280, 282,
# 283]  # 入库流量 (立方米/秒)
Q = [63, 89, 533, 339, 407, 934, 879, 596, 758, 1353, 1555, 1692, 2720, 3827, 3389,2286, 2501, 2031, 3922, 6966, 8912, 10969, 14016, 9919, 9802, 6609, 5672, 5101, 4072, 3348, 2994, 2194, 2161, 2044, 1794, 1577, 1168, 1054, 1098, 1402, 1329, 1059, 841, 759, 847, 876, 685, 701, 628, 555, 482, 423, 380, 383, 385, 390, 393, 396, 399, 403, 406, 409, 412, 413]

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
                if z_min <= z_curr <= z_max and 20 <= outflow <= max_outflow:
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
z_min = 191.24  # 最低水位
z_max = 202.8  # 最高水位
z0 = 191.24  # 初始水位

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

# 设置 Matplotlib 使用中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体或其他支持的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
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

plt.xlabel("时间段（3h）")
plt.ylabel("Optimal Outflows (m³/s)")
plt.legend()
plt.grid()
plt.show()
