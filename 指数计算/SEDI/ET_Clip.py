import os
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =========================================================
# 1. 路径设置
# =========================================================
input_dir = r"E:\data\ET_China"              # 已裁剪好的中国区域 ET 文件
output_dir = r"E:\data\ET_China_Clean"       # 清理负值后的输出目录

os.makedirs(output_dir, exist_ok=True)

start_year = 2001
end_year = 2022

# 是否每年随机绘图检查
PLOT_RANDOM_DAY = True

# =========================================================
# 2. 绘图函数
# =========================================================
def plot_random_day(da, year, var_name):
    print("\n" + "-" * 90)
    print(f"🖼️ {year} 年随机抽样绘图检查")
    print("-" * 90)

    # 自动识别经纬度名
    lat_candidates = ["lat", "latitude", "Latitude", "y"]
    lon_candidates = ["lon", "longitude", "Longitude", "x"]

    lat_name = None
    lon_name = None

    for c in lat_candidates:
        if c in da.coords:
            lat_name = c
            break

    for c in lon_candidates:
        if c in da.coords:
            lon_name = c
            break

    if lat_name is None or lon_name is None:
        print("⚠️ 未识别到经纬度坐标，跳过绘图。")
        return

    if "time" in da.dims:
        ntime = da.sizes["time"]
        if ntime == 0:
            print("⚠️ time 维为空，跳过绘图。")
            return

        rand_idx = random.randint(0, ntime - 1)
        rand_time = str(np.datetime_as_string(da["time"].values[rand_idx], unit="D"))
        print(f"随机时间索引: {rand_idx}")
        print(f"随机日期    : {rand_time}")

        da_plot = da.isel(time=rand_idx)
    else:
        rand_time = str(year)
        da_plot = da

    arr = da_plot.values
    valid_mask = np.isfinite(arr)

    print(f"有效像元数: {valid_mask.sum()}")
    if valid_mask.sum() > 0:
        print(f"最小值: {np.nanmin(arr):.4f}")
        print(f"最大值: {np.nanmax(arr):.4f}")
        print(f"均值  : {np.nanmean(arr):.4f}")

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(
        da_plot[lon_name].values,
        da_plot[lat_name].values,
        arr,
        shading="auto",
        cmap="viridis"
    )
    plt.colorbar(label=var_name)
    plt.title(f"{var_name} over China ({rand_time})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

# =========================================================
# 3. 批量处理
# =========================================================
failed_years = []

for year in range(start_year, end_year + 1):
    print("\n" + "=" * 90)
    print(f"🚀 开始处理 {year} 年")
    print("=" * 90)

    in_file = os.path.join(input_dir, f"ET.SiTHv2.A{year}_ChinaClip.nc")
    out_file = os.path.join(output_dir, f"ET.SiTHv2.A{year}_ChinaClip_clean.nc")

    if not os.path.exists(in_file):
        print(f"❌ 文件不存在，跳过: {in_file}")
        failed_years.append(year)
        continue

    try:
        ds = xr.open_dataset(in_file)

        # 自动识别主变量
        data_vars = list(ds.data_vars)
        if len(data_vars) == 0:
            raise ValueError("未找到数据变量。")
        var_name = data_vars[0]

        da = ds[var_name].astype("float32")

        print(f"变量名: {var_name}")
        print(f"数据维度: {da.dims}")
        print(f"数据形状: {da.shape}")

        # =========================
        # 处理前统计
        # =========================
        arr_before = da.values
        neg_mask_before = np.isfinite(arr_before) & (arr_before < 0)

        neg_count_before = np.sum(neg_mask_before)
        valid_count_before = np.sum(np.isfinite(arr_before))

        print("\n📊 处理前统计")
        print(f"有效像元数      : {valid_count_before}")
        print(f"负值像元数      : {neg_count_before}")

        if valid_count_before > 0:
            print(f"最小值          : {np.nanmin(arr_before):.4f}")
            print(f"最大值          : {np.nanmax(arr_before):.4f}")
            print(f"均值            : {np.nanmean(arr_before):.4f}")

        # =========================
        # 负值赋为 NaN
        # =========================
        da_clean = da.where(da >= 0)

        # =========================
        # 处理后统计
        # =========================
        arr_after = da_clean.values
        neg_mask_after = np.isfinite(arr_after) & (arr_after < 0)

        neg_count_after = np.sum(neg_mask_after)
        valid_count_after = np.sum(np.isfinite(arr_after))

        print("\n📊 处理后统计")
        print(f"有效像元数      : {valid_count_after}")
        print(f"负值像元数      : {neg_count_after}")

        if valid_count_after > 0:
            print(f"最小值          : {np.nanmin(arr_after):.4f}")
            print(f"最大值          : {np.nanmax(arr_after):.4f}")
            print(f"均值            : {np.nanmean(arr_after):.4f}")

        print(f"\n本次清理掉的负值像元数: {neg_count_before - neg_count_after}")

        # =========================
        # 保存结果
        # =========================
        ds_out = da_clean.to_dataset(name=var_name)
        ds_out[var_name].attrs.update(da.attrs)
        ds_out[var_name].attrs["note"] = "All negative values were set to NaN after China clipping."
        ds_out[var_name].attrs["dtype_saved"] = "float32"

        encoding = {
            var_name: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan)
            }
        }

        # 坐标变量保留默认保存方式
        for c in ds_out.coords:
            if c not in encoding:
                encoding[c] = {}

        ds_out.to_netcdf(out_file, encoding=encoding)

        print(f"\n✅ 已保存: {out_file}")

        # =========================
        # 随机绘图检查
        # =========================
        if PLOT_RANDOM_DAY:
            try:
                plot_random_day(da_clean, year, var_name)
            except Exception as e:
                print(f"⚠️ 绘图检查失败，但不影响主流程: {e}")

        ds.close()
        ds_out.close()

        print(f"✅ {year} 年处理完成")

    except Exception as e:
        print(f"❌ {year} 年处理失败: {e}")
        failed_years.append(year)

# =========================================================
# 4. 汇总
# =========================================================
print("\n" + "=" * 90)
print("🎯 全部处理完成")
print("=" * 90)

if len(failed_years) == 0:
    print("✅ 2001–2022 年全部处理成功")
else:
    print("⚠️ 以下年份处理失败或文件缺失：")
    print(failed_years)