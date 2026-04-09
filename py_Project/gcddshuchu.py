from Reservoir_Rulescheduling import *
import pandas as pd

# 读取Excel文件
file_path = r'C:\Users\mi\Desktop\水库设计洪水.xlsx'
sheet_name1 = '横山岭'
sheet_name2 = '龙门'
sheet_name3 = '王快'
sheet_name4 = '西大洋'
sheet_name5 = '区间入流'

# 使用 pandas 读取指定的 Excel 表格
df1 = pd.read_excel(file_path, sheet_name=sheet_name1)


l1 = 1
l2 = 2
l3 = 3

# 提取第一列数据
Q1 = df1.iloc[:, l1].tolist()  # `iloc[:, 0]` 选择第一列                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ，`tolist()` 将其转换为列表
Q1 = np.round(Q1, 2)
Q2 = df1.iloc[:, l2].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q2 = np.round(Q2, 2)
Q3 = df1.iloc[:, l3].tolist()  # `iloc[:, 0]` 选择第一列，`tolist()` 将其转换为列表
Q3 = np.round(Q3, 2)

# 输出结果
# print(Q1)
# print(Q2)
# print(Q3)
# print(Q4)

z1_list, q1_list = hsl_reservoir(Q1)
z2_list, q2_list = hsl_reservoir(Q2)
z3_list, q3_list = hsl_reservoir(Q3)


# 将列表组合成字典
data = {
    'q1': q1_list,
    'z1': z1_list,
    'q2': q2_list,
    'z2': z2_list,
    'q3': q3_list,
    'z3': z3_list
}

# 将字典转换为 DataFrame
df = pd.DataFrame(data)

# 指定保存路径
file_path = r'C:\Users\mi\Desktop\output.xlsx'

# 将 DataFrame 写入 Excel 文件
df.to_excel(file_path, index=False, sheet_name='Sheet1')

print(f"文件已保存到: {file_path}")