import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ================= 配置区 =================
# 请确保文件路径正确
file_path = r'E:\data\yeild_data\作物产量数据2000-2022.xlsx'
# output_path = r'C:\Users\wangrui\Desktop\文献\农业干旱指数引言\二次项去趋势标准化产量(SYRS).xlsx'
# sheets = ['小麦', '玉米', '水稻']
output_path = r'E:\data\计算结果\表格\二次项去趋势标准化产量(SYRS).xlsx'
sheets = [ '小麦','玉米','水稻']


def calculate_syrs(df_raw):
    """
    执行二次项去趋势并计算标准化产量残差 (SYRS)
    """
    # 1. 预处理：设置省份为索引，年份转为整数并升序排列
    df = df_raw.set_index(df_raw.columns[0])
    df.columns = df.columns.astype(int)
    df = df.sort_index(axis=1, ascending=True)

    years = df.columns.values.reshape(-1, 1)
    # 准备二次项特征 [1, t, t^2]
    poly = PolynomialFeatures(degree=2)
    years_poly = poly.fit_transform(years)

    # 初始化结果 DataFrame
    syrs_df = pd.DataFrame(index=df.index, columns=df.columns)

    # 2. 逐省份计算
    for province in df.index:
        y = df.loc[province].values.astype(float)

        # 排除缺失值 (NaN)
        mask = ~np.isnan(y)
        if mask.sum() > 5:  # 确保至少有6年数据进行二次拟合
            # 拟合二次模型
            model = LinearRegression()
            model.fit(years_poly[mask], y[mask])

            # 计算趋势产量 (Trend Yield)
            y_trend = model.predict(years_poly)

            # 计算残差 (Residuals) = 实际产量 - 趋势产量
            residuals = y - y_trend

            # 3. 标准化 (SYRS) = (残差 - 残差均值) / 残差标准差
            # 论文中通常对整个序列进行标准化，以消除不同省份产量的波动差异
            res_mean = np.nanmean(residuals)
            res_std = np.nanstd(residuals)

            if res_std != 0:
                syrs = (residuals - res_mean) / res_std
                syrs_df.loc[province] = syrs
            else:
                syrs_df.loc[province] = np.nan
        else:
            syrs_df.loc[province] = np.nan

    return syrs_df


# ================= 主程序 =================
try:
    print("开始处理数据...")
    with pd.ExcelWriter(output_path) as writer:
        for name in sheets:
            # 读取原始 Excel
            raw_data = pd.read_excel(file_path, sheet_name=name)

            # 计算 SYRS
            result = calculate_syrs(raw_data)

            # 写入新的工作表
            result.to_excel(writer, sheet_name=name)
            print(f"--- {name} 处理完成 ---")

    print(f"\n全部计算成功！文件已保存至：\n{output_path}")

except Exception as e:
    print(f"运行出错：{e}")
    print("提示：请检查 Excel 文件是否被其他程序占用，或 openpyxl 库是否安装。")