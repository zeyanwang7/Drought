import numpy as np
import matplotlib.pyplot as plt

z = [191,192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
     215]  # 王快水库水位
v = [3.733,4.03, 4.329, 4.644, 4.993, 5.301, 5.673, 6.049, 6.434, 6.834, 7.241, 7.66, 8.097, 8.575, 9.057, 9.549, 10.05,
     10.57, 11.136, 11.708, 12.288, 12.879, 13.484, 14.134, 14.789]  # 王快水库水位对应库容
q = [235,238, 451, 860, 1401, 2047, 2785, 3600, 4487, 5361, 6221, 7115, 8094, 9112, 10172, 11270, 12405, 13576, 14782,
     16020, 17294, 18599, 19936, 21303, 22701]  # 王快水库下泄流量能力
Q = [43, 61, 365, 232, 279, 640, 602, 408, 519, 927, 1065, 1159, 1863, 2621, 2321, 1566, 1713, 1391, 2686, 4771, 6104,
     7513, 9600, 6794, 6714, 4527, 3885, 3494, 2787, 2293, 2051, 1503, 1480, 1400, 1229, 1080, 800, 722, 752, 900, 910,
     725, 576, 520, 580, 600, 469, 480, 430, 380, 330, 290, 260, 262, 264, 267, 269, 270, 273, 276, 278, 280,
     282, 283]  # 王快水库入库流量


# 线性插值函数
def linear_interpolate(x, y, x_new):
    if x_new < x[0] or x_new > x[-1]:
        raise ValueError("Interpolation input is out of range!")
    return np.interp(x_new, x, y)


# 目标函数
def objective_function(zx):
    zx[0] = 191.24
    vs = []  # 初始化库容序列
    qs = [43]  # 初始化下泄流量序列
    t = 3 * 60 * 60  # 时段长
    for i in range(len(zx)):
        vt = linear_interpolate(z, v, zx[i])
        vs.append(vt)

    for i in range(len(vs) - 1):
        qt = (Q[i] + Q[i + 1]) - (2 * (vs[i + 1] - vs[i]) * 100000000) / t - qs[i]
        qs.append(qt)

    q2_sum = sum(x ** 2 for x in qs)
    return q2_sum, qs


# 约束惩罚函数
def constraint_penalty(zx, qx, Q):
    zmin = 191.24
    zmax = 202.8
    penalty = 0.0
    qmax_list = []  # 下泄能力

    for i in range(len(zx)):  # 水位约束
        if zx[i] > zmax:
            penalty += 1e8
        if zx[i] < zmin:
            penalty += 1e8
        qmax = linear_interpolate(z, q, zx[i])
        qmax_list.append(qmax)

    if zx[0] != zmin:  # 初水位约束
        penalty += 1e8

    for i in range(len(qx)):  # 下泄流量约束
        if qx[i] > qmax_list[i]:
            qx[i] = qmax_list[i]
        elif qx[i] <= 0:
            qx[i] = 100

    for i in range(len(qx)-1):  # 泄流变幅约束
        if abs(qx[i] - qx[i+1]) >= 500:
            penalty += 1e8

    for i in range(len(qx) - 1):  # 水位修正
        v1 = linear_interpolate(z, v, zx[i])
        v2 = np.clip(v1 + ((Q[i] - qx[i]) * 60 * 60 * 3) / 100000000, v[0], v[-1])
        zx[i + 1] = linear_interpolate(v, z, v2)
    return penalty, qx, zx


# 粒子类
class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)  # 初始化位置
        self.velocity = np.random.uniform(-0.3, 0.3, dim)  # 初始化速度
        self.best_position = np.copy(self.position)  # 个体历史最优位置
        self.best_value = float('inf')  # 个体历史最优值
        self.value = float('inf')  # 当前目标函数值
        self.lower_bound = lower_bound  # 添加粒子的下界属性
        self.upper_bound = upper_bound  # 添加粒子的上界属性

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.lower_bound, self.upper_bound)  # 边界约束

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (
                global_best_position - self.position)

        # 限制速度范围
        v_max = (self.upper_bound - self.lower_bound) / 2
        v_min = -v_max
        self.velocity = np.clip(self.velocity, v_min, v_max)


# 粒子群优化（PSO）实现
def pso_with_constraints(objective_function, constraint_penalty, num_particles=30, max_iter=500, dim=64,
                         lower_bound=193, upper_bound=198):
    particles = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
    w = 0.5  # 惯性权重
    c1 = 1.2  # 个人经验因子
    c2 = 1.2  # 社会经验因子

    global_best_position = None
    global_best_value = float('inf')
    fitness_history = []  # 记录每次迭代的全局最优适应度值
    global_best_qs = []

    for t in range(max_iter):

        for particle in particles:
            obj_value, qs = objective_function(particle.position)  # 解包目标函数返回值

            # 调用约束函数，获取修正后的流量和位置
            penalty, qs, zx = constraint_penalty(particle.position, qs, Q)

            # 更新粒子的位置为修正后的值
            particle.position = zx

            # 更新适应度值
            particle.value = obj_value + penalty

            # 更新个体最优
            if particle.value < particle.best_value:
                particle.best_value = particle.value
                particle.best_position = np.copy(particle.position)

            # 更新全局最优
            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = np.copy(particle.best_position)
                global_best_qs = qs  # 更新最佳流量过程

        # 更新速度和位置
        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

        # 记录全局最优值
        fitness_history.append(global_best_value)

        if (t + 1) % 10 == 0:  # 每10次迭代输出结果
            print(f"Iteration {t + 1}/{max_iter}, Best Value: {global_best_value}")

    # 绘制适应度值变化折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), fitness_history, marker='o', color='b', label='Fitness Value')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Value Over Iterations')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制流量过程折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(global_best_qs)), global_best_qs, marker='o', color='g', label='Discharge (qs)')
    plt.plot(range(len(Q)), Q, marker='o', color='r', label='Inflow (Q)')
    plt.xlabel('Time Step')
    plt.ylabel('Discharge (qs)')
    plt.title('Discharge Process')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制水位变化折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(dim), global_best_position, marker='o', color='r', label='Z')
    plt.xlabel('Iteration')
    plt.ylabel('Best Position')
    plt.title('Best Position Over Iterations')
    plt.legend()
    plt.grid()
    plt.show()

    return global_best_position, global_best_value


# 运行PSO优化
best_position, best_value = pso_with_constraints(objective_function, constraint_penalty, num_particles=20, max_iter=100,
                                                 dim=64, lower_bound=193, upper_bound=198)

print("Optimized Position:", best_position)
print("Optimized Value:", best_value)
