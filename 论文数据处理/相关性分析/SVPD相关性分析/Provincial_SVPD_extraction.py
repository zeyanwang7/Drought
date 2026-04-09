import xarray as xr
import pandas as pd
import regionmask
import geopandas as gpd
import os
import numpy as np
#该代码用于计算各省SVPD平均值
# --- 1. 路径设置 ---
# 使用我们之前生成的 SVPD 结果文件
# svpd_nc_path = r'E:\data\meteorologicaldata\IndexCalculationResults\SVPD\SVPD_010deg_8day_Wheat_2000_2022.nc'
# shp_path = r'E:\数据集\中国_省\中国_省2.shp'
# output_csv = r'E:\data\计算结果\Provinces_Wheat_SVPD_Steps_2000_2022.csv'

svpd_nc_path = r'E:\data\meteorologicaldata\IndexCalculationResults\SVPD\SVPD_010deg_8day_Cotton_2000_2022.nc'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
output_csv = r'E:\data\计算结果\Provinces_Cotton_SVPD_Steps_2000_2022.csv'

# --- 2. 加载数据 ---
print("正在读取 SVPD 数据...")
ds = xr.open_dataset(svpd_nc_path)
# 确保变量名正确，根据之前脚本定义的变量名为 'SVPD'
svpd = ds['SVPD']

# --- 3. 加载并清洗 SHP ---
print("正在读取省份边界并执行预处理...")
try:
    provinces_gdf = gpd.read_file(shp_path, encoding='utf-8')
    # 简单的编码检查
    if 'name' in provinces_gdf.columns and provinces_gdf['name'].iloc[0] is None:
        raise ValueError
except:
    provinces_gdf = gpd.read_file(shp_path, encoding='gbk')

# 统一坐标系为 WGS84 (EPSG:4326) 以匹配 NC 文件
provinces_gdf = provinces_gdf.to_crs(epsg=4326)
provinces_gdf['geometry'] = provinces_gdf['geometry'].make_valid()

# --- 4. 执行空间提取 ---
print("开始按省份提取作物种植区 SVPD 空间均值...")
results = {}

# 获取 NC 文件的经纬度，用于生成掩膜
lon = svpd.lon
lat = svpd.lat

for index, row in provinces_gdf.iterrows():
    # 自动识别名称列（可能是 'name' 或 '省' 等）
    prov_name = row.get('name') or row.get('省') or f"Prov_{index}"
    geom = row['geometry']

    try:
        # 使用 regionmask 针对当前省份几何体生成掩膜
        # mask_geopandas 会返回一个与 lon/lat 形状相同的阵列，省内为 0，省外为 NaN
        my_mask = regionmask.mask_geopandas(gpd.GeoSeries([geom]), lon, lat)

        # 核心逻辑：
        # 1. .where(my_mask == 0) 只保留该省份范围内的数据
        # 2. 由于 SVPD 原本就只有作物区有值，这里计算出的 mean 自动就是“省内作物区均值”
        prov_series = svpd.where(my_mask == 0).mean(dim=("lat", "lon")).to_series()

        # 检查是否提取到了有效数据
        if not prov_series.dropna().empty:
            results[prov_name] = prov_series
            print(f"  - [成功] {prov_name} (包含有效作物区数据)")
        else:
            print(f"  - [跳过] {prov_name} (该省范围内无作物种植区数据)")

    except Exception as e:
        print(f"  - [失败] {prov_name} 错误信息: {e}")

# --- 5. 组装与清洗 ---
print("\n正在组装数据表格...")
df = pd.DataFrame(results)

# 按照时间排序
df = df.sort_index()

# 剔除全空的列（完全没有水稻的省份）
df_final = df.dropna(axis=1, how='all')

# --- 6. 导出结果 ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
# 使用 utf-8-sig 以确保 Excel 打开中文不乱码
df_final.to_csv(output_csv, encoding='utf-8-sig')

print("-" * 30)
print(f"✅ 处理完成！")
print(f"结果保存路径: {output_csv}")
print(f"共提取有效省份数量: {len(df_final.columns)}")
print(f"有效省份列表: {df_final.columns.tolist()[:10]} ... (等)")