import os
import xarray as xr
from pathlib import Path
import pandas as pd
##降水数据重采样到8天尺度
# --- 1. 路径设置 ---
base_dir = Path(r"E:\data\meteorologicaldata\precipitation")
output_dir = Path(r"E:\data\meteorologicaldata\Precip_8Day_Standard")
output_dir.mkdir(parents=True, exist_ok=True)

# --- 2. 循环处理 2020-2022 年 ---
target_years = [2020, 2021, 2022]

for year in target_years:
    year_dir = base_dir / str(year)

    if not year_dir.exists():
        print(f"跳过：找不到 {year} 年目录 {year_dir}")
        continue

    # 获取该年份下所有的 .nc 文件
    file_paths = sorted(list(year_dir.glob("*.nc")))

    # 正常年份应有 365 或 366 个文件
    if len(file_paths) < 360:
        print(f"警告：{year} 年文件数量显著不足（仅 {len(file_paths)} 个）")

    print(f"正在读取 {year} 年数据，共 {len(file_paths)} 个文件...")

    datasets = []
    for p in file_paths:
        try:
            # 根据文件名解析日期，例如 ChinaMet_010deg_prec_2020_01_01.nc
            # 取文件名最后三个部分作为年、月、日
            date_parts = p.stem.split('_')[-3:]
            current_date = pd.to_datetime("-".join(date_parts))

            # 打开数据集并添加时间维度
            ds = xr.open_dataset(p)
            ds = ds.expand_dims(time=[current_date])
            datasets.append(ds)
        except Exception as e:
            print(f"处理文件 {p.name} 时出错: {e}")

    if datasets:
        # 1. 沿时间维度合并文件
        year_ds = xr.concat(datasets, dim="time")

        # 2. 8天尺度重采样
        # .sum() 会将每8天内的降水量进行累加
        # label="left" 确保 8 天周期的起始日期作为标记
        year_8day = year_ds.resample(time="8D", label="left").sum()

        # 3. 保存结果
        file_name = f"ChinaMet_0.10deg_Precip_8Day_{year}.nc"
        output_path = output_dir / file_name

        # 推荐显式指定计算引擎
        year_8day.to_netcdf(output_path)

        print(f"--- {year} 年处理完成，已生成：{file_name} (共 {len(year_8day.time)} 个时段) ---")

        # 及时关闭数据集释放内存
        year_ds.close()
        for d in datasets:
            d.close()
    else:
        print(f"{year} 年没有有效的数据集可处理。")

print("\n✅ 2020-2022 年降水数据 8 天尺度序列化处理已全部完成。")