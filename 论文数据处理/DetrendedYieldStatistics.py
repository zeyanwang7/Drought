import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 配置区 ---
file_path = r'C:\Users\wangrui\Desktop\文献\农业干旱指数引言\作物产量数据2000-2022.xlsx'
sheets = ['小麦', '玉米', '水稻']
output_sorted = '产量数据_升序排列.xlsx'
output_detrended = '去趋势产量结果.xlsx'

# 预准备一个用于存储升序原始数据的字典
sorted_dfs = {}
# 预准备一个用于存储去趋势结果的字典
detrended_dfs = {}

# --- 处理逻辑 ---
for name in sheets:
    # 1. 读取数据
    # header=0 表示第一行是年份，index_col=0 表示第一列是省份
    df = pd.read_excel(file_path, sheet_name=name, index_col=0)

    # 2. 年份列名转为整数并升序排列 (2000 -> 2022)
    df.columns = df.columns.astype(int)
    df_sorted = df.sort_index(axis=1, ascending=True)
    sorted_dfs[name] = df_sorted  # 存入升序字典

    # 3. 去趋势计算
    years = df_sorted.columns.values.reshape(-1, 1)
    detrended_data = pd.DataFrame(index=df_sorted.index, columns=df_sorted.columns)

    for province in df_sorted.index:
        y = df_sorted.loc[province].values

        # 掩码：处理可能存在的缺失值 (NaN)
        mask = ~np.isnan(y)
        if mask.sum() > 5:  # 至少有5个有效数据点才进行回归
            model = LinearRegression()
            model.fit(years[mask], y[mask])

            # 计算线性趋势值线
            trend = model.predict(years)

            # 计算 RYD = (观测值 - 趋势值) / 趋势值
            # 这样得到的是一个比例，反映了产量偏离正常技术水平的程度
            ryd = (y - trend) / trend
            detrended_data.loc[province] = ryd
        else:
            detrended_data.loc[province] = np.nan

    detrended_dfs[name] = detrended_data

# --- 写入 Excel ---

# 保存升序排列后的原始数据
with pd.ExcelWriter(output_sorted) as writer1:
    for name in sheets:
        sorted_dfs[name].to_excel(writer1, sheet_name=name)

# 保存去趋势后的结果
with pd.ExcelWriter(output_detrended) as writer2:
    for name in sheets:
        detrended_dfs[name].to_excel(writer2, sheet_name=name)

print(f"处理完成！")
print(f"1. 升序排列文件已保存至: {output_sorted}")
print(f"2. 去趋势结果文件已保存至: {output_detrended}")