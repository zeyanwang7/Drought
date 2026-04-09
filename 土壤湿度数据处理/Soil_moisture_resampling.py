# -*- coding: utf-8 -*-
"""
将 10/20/30/.../100 cm 土壤湿度重组为：
0-10 cm, 10-40 cm, 40-80 cm, 0-80 cm

处理规则：
- 0-10 cm  : 使用 10cm
- 10-40 cm : 平均 10,20,30,40cm
- 40-80 cm : 平均 40,50,60,70,80cm
- 0-80 cm  : 平均 10,20,30,40,50,60,70,80cm
"""

import os
import re
import numpy as np
import xarray as xr
from glob import glob

# =========================
# 1. 路径设置
# =========================
base_dir = r"E:\data\Soil_Humidity"
out_dir = r"E:\data\Soil_Humidity_ReSample"

# 处理年份范围，可自行修改
start_year = 2000
end_year = 2022

# =========================
# 2. 新深度层定义
# =========================
layer_definitions = {
    "0_10cm":  [10],
    "10_40cm": [10, 20, 30, 40],
    "40_80cm": [40, 50, 60, 70, 80],
    "0_80cm":  [10, 20, 30, 40, 50, 60, 70, 80],
}

# 原始深度目录
source_depths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# =========================
# 3. 工具函数
# =========================
def find_nc_file(depth_cm, year):
    """
    根据深度和年份查找对应 nc 文件
    例如：E:\\data\\Soil_Humidity\\10cm\\SMCI_9km_2000_10cm.nc
    """
    folder = os.path.join(base_dir, f"{depth_cm}cm")
    pattern = os.path.join(folder, f"SMCI_9km_{year}_{depth_cm}cm.nc")
    files = glob(pattern)
    if len(files) == 0:
        return None
    return files[0]


def detect_main_variable(ds):
    """
    自动识别主变量名
    优先排除常见坐标变量，取真正的数据变量
    """
    exclude_names = {"lat", "latitude", "lon", "longitude", "time", "depth"}
    data_vars = [v for v in ds.data_vars if v.lower() not in exclude_names]

    if len(data_vars) == 1:
        return data_vars[0]
    elif len(data_vars) > 1:
        # 优先挑 3 维或 2 维变量
        for v in data_vars:
            if ds[v].ndim >= 2:
                return v
        return data_vars[0]
    else:
        raise ValueError("未识别到有效数据变量，请检查 nc 文件内容。")


def load_dataarray(file_path):
    """
    读取 nc 文件，并返回主数据变量对应的 DataArray
    """
    ds = xr.open_dataset(file_path)
    var_name = detect_main_variable(ds)
    da = ds[var_name]
    return ds, da, var_name


def build_output_encoding(da, var_name):
    """
    输出压缩设置
    """
    encoding = {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    return encoding


def ensure_same_coords(dataarrays, depths):
    """
    检查多层数据的维度和坐标是否一致
    """
    ref = dataarrays[0]
    for i, da in enumerate(dataarrays[1:], start=1):
        if da.dims != ref.dims:
            raise ValueError(f"维度不一致：{depths[0]}cm 与 {depths[i]}cm 的 dims 不同")
        for dim in ref.dims:
            if dim in da.coords and dim in ref.coords:
                if da[dim].shape != ref[dim].shape:
                    raise ValueError(f"坐标长度不一致：{dim}")
    return True


# =========================
# 4. 主处理函数
# =========================
def process_one_year(year):
    print("=" * 100)
    print(f"开始处理年份: {year}")
    print("=" * 100)

    # 先检查该年各深度文件是否存在
    file_map = {}
    for d in source_depths:
        fp = find_nc_file(d, year)
        file_map[d] = fp
        if fp is None:
            print(f"⚠️ 缺少文件: {d}cm, {year}")
        else:
            print(f"✅ 找到文件: {os.path.basename(fp)}")

    # 逐个目标层处理
    for out_layer_name, need_depths in layer_definitions.items():
        print("\n" + "-" * 80)
        print(f"处理目标层: {out_layer_name}  <-  {need_depths}")
        print("-" * 80)

        # 检查是否缺文件
        missing_depths = [d for d in need_depths if file_map[d] is None]
        if len(missing_depths) > 0:
            print(f"❌ 跳过 {out_layer_name}，缺少深度文件: {missing_depths}")
            continue

        datasets = []
        dataarrays = []
        var_names = []

        try:
            # 读取各深度数据
            for d in need_depths:
                ds, da, var_name = load_dataarray(file_map[d])
                datasets.append(ds)
                dataarrays.append(da)
                var_names.append(var_name)
                print(f"  已读取 {d}cm | 变量名: {var_name} | shape: {da.shape}")

            # 检查变量名是否一致
            if len(set(var_names)) != 1:
                raise ValueError(f"变量名不一致: {var_names}")

            var_name = var_names[0]

            # 检查坐标是否一致
            ensure_same_coords(dataarrays, need_depths)

            # 堆叠后求平均
            stacked = xr.concat(dataarrays, dim="new_depth_stack")
            mean_da = stacked.mean(dim="new_depth_stack", skipna=True)

            # 添加属性说明
            mean_da.attrs = dataarrays[0].attrs.copy()
            mean_da.attrs["description"] = f"Soil humidity averaged for {out_layer_name.replace('_', '-')}"
            mean_da.attrs["source_depths_cm"] = ",".join(map(str, need_depths))
            mean_da.attrs["processing_method"] = "mean of included depth layers"

            # 转成 Dataset
            out_ds = mean_da.to_dataset(name=var_name)

            # 增加全局属性
            out_ds.attrs["title"] = f"Soil humidity for {out_layer_name.replace('_', '-')}"
            out_ds.attrs["year"] = str(year)
            out_ds.attrs["source_depths_cm"] = ",".join(map(str, need_depths))
            out_ds.attrs["method"] = "simple arithmetic mean"

            # 输出目录
            save_folder = os.path.join(out_dir, out_layer_name)
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, f"SMCI_9km_{year}_{out_layer_name}.nc")

            encoding = build_output_encoding(mean_da, var_name)
            out_ds.to_netcdf(save_path, encoding=encoding)

            print(f"✅ 输出完成: {save_path}")

        except Exception as e:
            print(f"❌ 处理失败: {out_layer_name}, year={year}, error={e}")

        finally:
            # 关闭数据集
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass


def main():
    os.makedirs(out_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        process_one_year(year)

    print("\n全部处理完成！")


if __name__ == "__main__":
    main()