from matplotlib import pyplot as plt
from Reservoir_Rulescheduling import *
import pandas as pd


def linear_interpolate(x, y, x_new):  # 线性插值
    return np.interp(x_new, x, y)


def wk_reservoir(Q):
    z = [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
         214, 215]
    v = [4.03, 4.329, 4.644, 4.993, 5.301, 5.673, 6.049, 6.434, 6.834, 7.241, 7.66, 8.097, 8.575, 9.057, 9.549, 10.05,
         10.57, 11.136, 11.708, 12.288, 12.879, 13.484, 14.134, 14.789]
    q = [238, 451, 860, 1401, 2047, 2785, 3600, 4487, 5361, 6221, 7115, 8094, 9112, 10172, 11270, 12405, 13576, 14782,
         16020, 17294, 18599, 19936, 21303, 22701]
    zs = 191.24
    qs = Q[0]
    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])
        if 191.24 <= z_list[-1] < 197.53:
            if qt > 800:
                qt = 800
        if 197.53 <= z_list[-1] < 199.37:
            if qt > 2500:
                qt = 2500
        if 199.37 <= z_list[-1] < 202.8:
            if qt > 7000:
                qt = 7000

        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qt, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list


# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\水库设计洪水.xlsx'

sheet_name2 = '王快'

# 使用 pandas 读取指定的 Excel 表格

df2 = pd.read_excel(file_path, sheet_name=sheet_name2)

ll = 1
wx = 0
# 提取第一列数据

Q2 = df2.iloc[:, ll].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)

z2_list, q2_list = lm_reservoir(Q2)
out_q = muskingum_non_linear(q2_list, 14, 0.29, 14, 00, 1)
# 输出结果
print(out_q)

# 创建 x 轴数据（索引）
x = range(len(q2_list))

# 绘制折线图
plt.figure(figsize=(8, 5))  # 设置画布大小
plt.plot(x, q2_list, label='q2_list', marker='o', linestyle='-', color='blue')  # 绘制 q2_list
plt.plot(x, out_q, label='out_q', marker='s', linestyle='--', color='red')  # 绘制 out_q

# 添加标题和标签
plt.title('Line Chart of q2_list and out_q')  # 图表标题
plt.xlabel('Index')  # x 轴标签
plt.ylabel('Value')  # y 轴标签

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

# 将列表组合成字典
data = {
    'inflow': q2_list,
    'outflow': out_q,
}

# 将字典转换为 DataFrame
df = pd.DataFrame(data)

# 指定保存路径
file_path = r'C:\Users\mi\Desktop\output27.xlsx'

# 将 DataFrame 写入 Excel 文件
df.to_excel(file_path, index=False, sheet_name='王快洪水演进20年')

print(f"文件已保存到: {file_path}")
