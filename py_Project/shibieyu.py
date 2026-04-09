import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
z = [724, 725, 726, 727, 728, 729, 730, 731, 732, 732.5]
v = [0.2, 0.205, 0.213, 0.221, 0.229, 0.237, 0.245, 0.253, 0.26, 0.264]
q = [184.4, 408.7, 485.2, 561.7, 638.2, 714.7, 792, 833.2, 874.4, 894.9]
qin = [16.86, 82.72, 285.19, 307.59, 368, 285.19, 218.98, 152.78, 101.85, 82.72, 74.95, 60.89, 42.16]

t = [0, 1, 2, 3,4, 6, 8, 10, 12, 13, 16, 20, 24]

# 要插值的目标时间点
t_target = np.arange(0, 25)  # 目标时间点 t = [1,2,3,...,24]

# 使用线性插值（默认）进行插值
interpolator = interp1d(t, qin, kind='linear', fill_value="extrapolate")

# 计算对应的q值
q_target = interpolator(t_target)

# 输出插值结果
print("插值后的q值：", q_target)
qin = q_target
def linear_interpolate(x, y, x_new):  # 线性插值
    return np.interp(x_new, x, y)


zs = 725
qs = qin[0]
vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
v_list = [vs]
z_list = [zs]
q_list = [qs]

for i in range(len(qin)):
    qt = linear_interpolate(z, q, z_list[-1])
    if i != len(qin)-1:
        if z_list[-1] > 725:
            qt = 120
        else:
            qt = qin[i+1]
            if qt > 120:
                qt = 120
        vt = (qin[i] + qin[i + 1]) * 1 * 60 * 60 / 2 - (q_list[-1] + qt) * 1 * 60 * 60 / 2 + v_list[-1] * 100000000
    else:
        if z_list[-1] > 725:
            qt = 120
        else:
            qt = qin[-1]
            if qt > 120:
                qt = 120
        vt = (qin[-2] + qin[-1]) * 1 * 60 * 60 / 2 - (q_list[-1] + qt) * 1 * 60 * 60 / 2 + v_list[-1] * 100000000
    v2 = vt / 100000000
    zt = linear_interpolate(v, z, v2)
    q_list.append(round(qt, 1))
    v_list.append(round(v2, 3))
    z_list.append(round(zt, 2))
print(qin, len(qin))
print(q_list, len(q_list))
print(v_list, len(v_list))
print(z_list, len(z_list))
q_list.pop()
v_list.pop()
z_list.pop()
# 假设qin, q_list, v_list, z_list 是你的数据列表

# 将四个列表合并成一个 DataFrame
df = pd.DataFrame({
    '入库流量（m3/s）': qin,
    '下泄流量（m3/s）': q_list,
    '库容(亿m3)': v_list,
    '水位(m)': z_list
})

# 使用ExcelWriter指定文件和工作表
file_path = r'C:\Users\mi\Desktop\shibeiyu.xlsx'  # 文件路径
sheet_name = '20年一遇洪水'  # 工作表名称

# 将 DataFrame 写入 Excel
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)  # index=False 表示不保存行索引

