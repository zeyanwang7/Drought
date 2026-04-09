import os
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt

# =========================================================
# 1. 基本参数设置
# =========================================================
sedi_base_dir = r"E:\data\SEDI\Wheat"
output_base_dir = r"E:\data\SEDI\Wheat\Pheno_Stage_Mean"

os.makedirs(output_base_dir, exist_ok=True)

crops = ["Spring_Wheat", "Winter_Wheat"]
scales = [1, 3, 6, 9, 12]   # 如果你目前只有部分尺度，可以自行删减

start_date = "2002-01-01"
end_date   = "2022-12-31"


# =========================================================
# 2. 工具函数：自动识别 SEDI 变量名
# =========================================================
def detect_sedi_var(ds):
    candidates = list(ds.data_vars)

    for v in candidates:
        if "sedi" in v.lower():
            return v

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"无法自动识别 SEDI 变量名，当前变量有: {candidates}")


# =========================================================
# 3. 工具函数：读取并标准化 SEDI 数据
# =========================================================
def prepare_sedi_dataset(sedi_path):
    ds = xr.open_dataset(sedi_path)

    print("=" * 100)
    print(f"📂 读取 SEDI 文件: {sedi_path}")
    print("=" * 100)
    print(ds)

    var_name = detect_sedi_var(ds)
    print(f"\n✅ 自动识别 SEDI 变量名: {var_name}")

    da = ds[var_name]

    if not {"time", "lat", "lon"}.issubset(set(da.dims)):
        raise ValueError(f"SEDI 变量维度不是(time, lat, lon)体系，当前维度为: {da.dims}")

    da = da.transpose("time", "lat", "lon")

    # 纬度升序
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")

    # 时间截取
    da = da.sel(time=slice(start_date, end_date))

    print(f"\n📌 SEDI 时间范围: {str(da.time.values[0])[:10]} ~ {str(da.time.values[-1])[:10]}")
    print(f"📌 SEDI shape: {da.shape}")

    return ds, da, var_name


# =========================================================
# 4. 工具函数：读取 tif 并转为与 SEDI 对齐的 DataArray
# =========================================================
def read_tif_as_da(tif_path, target_lat=None, target_lon=None, name="pheno"):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        height = src.height
        width = src.width

        if nodata is not None:
            arr[arr == nodata] = np.nan

        # 常见无效值
        for invalid in [-9999, -999, 0]:
            arr[arr == invalid] = np.nan

        # 像元中心坐标
        x_coords = np.array(
            [transform.c + transform.a * (i + 0.5) for i in range(width)],
            dtype=np.float64
        )
        y_coords = np.array(
            [transform.f + transform.e * (j + 0.5) for j in range(height)],
            dtype=np.float64
        )

        # 若纬度降序则翻转
        if y_coords[0] > y_coords[-1]:
            y_coords = y_coords[::-1]
            arr = np.flipud(arr)

        da = xr.DataArray(
            arr,
            coords={"lat": y_coords, "lon": x_coords},
            dims=("lat", "lon"),
            name=name
        )

        # 对齐到 SEDI 网格
        if target_lat is not None and target_lon is not None:
            da = da.interp(lat=target_lat, lon=target_lon, method="nearest")

    return da


# =========================================================
# 5. 读取某作物的三个物候文件
# =========================================================
def get_pheno_paths(crop):
    if crop == "Spring_Wheat":
        spring_base_dir = r"E:\数据集\重采样后物候期\小麦\0.1度"
        return {
            "GR_EM": os.path.join(spring_base_dir, "Spring_Wheat_GR_EM_0.1deg.tif"),
            "HE":    os.path.join(spring_base_dir, "Spring_Wheat_HE_0.1deg.tif"),
            "MA":    os.path.join(spring_base_dir, "Spring_Wheat_MA_0.1deg.tif"),
        }

    elif crop == "Winter_Wheat":
        winter_base_dir = r"E:\数据集\重采样后物候期\小麦\0.1度_修复版"
        return {
            "GR_EM": os.path.join(winter_base_dir, "Winter_Wheat_GR_EM_0.1deg_fixed.tif"),
            "HE":    os.path.join(winter_base_dir, "Winter_Wheat_HE_0.1deg_fixed.tif"),
            "MA":    os.path.join(winter_base_dir, "Winter_Wheat_MA_0.1deg_fixed.tif"),
        }

    else:
        raise ValueError(f"不支持的作物类型: {crop}")


def load_pheno_for_crop(crop, target_lat, target_lon):
    paths = get_pheno_paths(crop)

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{crop} 的物候文件不存在: {path}")

    print("\n" + "-" * 100)
    print(f"📥 读取 {crop} 物候文件")
    print("-" * 100)
    for k, v in paths.items():
        print(f"{k}: {v}")

    gr_da = read_tif_as_da(paths["GR_EM"], target_lat=target_lat, target_lon=target_lon, name="GR_EM_DOY")
    he_da = read_tif_as_da(paths["HE"],    target_lat=target_lat, target_lon=target_lon, name="HE_DOY")
    ma_da = read_tif_as_da(paths["MA"],    target_lat=target_lat, target_lon=target_lon, name="MA_DOY")

    print(f"✅ {crop} 物候数据读取完成并已对齐到 SEDI 网格")
    print(f"GR&EM shape: {gr_da.shape}")
    print(f"HE    shape: {he_da.shape}")
    print(f"MA    shape: {ma_da.shape}")

    return gr_da, he_da, ma_da, paths


# =========================================================
# 6. 统计物候有效性
# =========================================================
def check_pheno_validity(crop, gr_da, he_da, ma_da):
    valid_all = np.isfinite(gr_da) & np.isfinite(he_da) & np.isfinite(ma_da)
    valid_triplet = valid_all & (gr_da < he_da) & (he_da < ma_da)

    valid_count = int(valid_all.sum().values)
    ok_count = int(valid_triplet.sum().values)

    print("\n" + "-" * 100)
    print(f"🔍 {crop} 物候有效性检查")
    print("-" * 100)
    print(f"三期同时有效像元数              : {valid_count}")
    print(f"满足 GR&EM < HE < MA 的像元数   : {ok_count}")
    print(f"不满足顺序的像元数              : {valid_count - ok_count}")

    if ok_count > 0:
        s1 = (he_da - gr_da).where(valid_triplet)
        s2 = (ma_da - he_da).where(valid_triplet)
        total = (ma_da - gr_da).where(valid_triplet)

        print(f"阶段1长度均值(HE-GR&EM): {float(s1.mean(skipna=True).values):.2f} 天")
        print(f"阶段2长度均值(MA-HE)  : {float(s2.mean(skipna=True).values):.2f} 天")
        print(f"总长度均值(MA-GR&EM)  : {float(total.mean(skipna=True).values):.2f} 天")


# =========================================================
# 7. 核心函数：计算逐年两个阶段平均 SEDI
# =========================================================
def calc_stage_mean_by_year(sedi_da, gr_da, he_da, ma_da):
    valid_pheno = (
        np.isfinite(gr_da) & np.isfinite(he_da) & np.isfinite(ma_da) &
        (gr_da < he_da) & (he_da < ma_da)
    )

    gr_da = xr.where(valid_pheno, np.rint(gr_da), np.nan)
    he_da = xr.where(valid_pheno, np.rint(he_da), np.nan)
    ma_da = xr.where(valid_pheno, np.rint(ma_da), np.nan)

    doy = xr.DataArray(
        sedi_da.time.dt.dayofyear.values,
        coords={"time": sedi_da.time},
        dims=("time",),
        name="doy"
    )

    doy_3d, gr_3d = xr.broadcast(doy, gr_da)
    _, he_3d = xr.broadcast(doy, he_da)
    _, ma_3d = xr.broadcast(doy, ma_da)
    _, valid_3d = xr.broadcast(doy, valid_pheno)

    # 两阶段掩膜
    stage1_mask = valid_3d & (doy_3d >= gr_3d) & (doy_3d < he_3d)
    stage2_mask = valid_3d & (doy_3d >= he_3d) & (doy_3d <= ma_3d)

    # 阶段内 SEDI
    sedi_stage1 = sedi_da.where(stage1_mask)
    sedi_stage2 = sedi_da.where(stage2_mask)

    # 按年求平均
    stage1_mean = sedi_stage1.groupby("time.year").mean(dim="time", skipna=True)
    stage2_mean = sedi_stage2.groupby("time.year").mean(dim="time", skipna=True)

    # 有效天数
    stage1_days = (stage1_mask & np.isfinite(sedi_da)).groupby("time.year").sum(dim="time")
    stage2_days = (stage2_mask & np.isfinite(sedi_da)).groupby("time.year").sum(dim="time")

    stage1_mean = stage1_mean.rename("SEDI_stage1_mean").astype(np.float32)
    stage2_mean = stage2_mean.rename("SEDI_stage2_mean").astype(np.float32)
    stage1_days = stage1_days.rename("SEDI_stage1_days").astype(np.int16)
    stage2_days = stage2_days.rename("SEDI_stage2_days").astype(np.int16)

    gr_da = gr_da.rename("GR_EM_DOY").astype(np.float32)
    he_da = he_da.rename("HE_DOY").astype(np.float32)
    ma_da = ma_da.rename("MA_DOY").astype(np.float32)
    valid_pheno = valid_pheno.rename("valid_pheno_mask").astype(np.int8)

    out_ds = xr.Dataset({
        "SEDI_stage1_mean": stage1_mean,
        "SEDI_stage2_mean": stage2_mean,
        "SEDI_stage1_days": stage1_days,
        "SEDI_stage2_days": stage2_days,
        "GR_EM_DOY": gr_da,
        "HE_DOY": he_da,
        "MA_DOY": ma_da,
        "valid_pheno_mask": valid_pheno
    })

    return out_ds


# =========================================================
# 8. 保存 nc
# =========================================================
def save_output_nc(ds_out, output_nc, crop, scale, sedi_path, pheno_paths):
    encoding = {
        "SEDI_stage1_mean": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "SEDI_stage2_mean": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "SEDI_stage1_days": {"zlib": True, "complevel": 4, "dtype": "int16"},
        "SEDI_stage2_days": {"zlib": True, "complevel": 4, "dtype": "int16"},
        "GR_EM_DOY": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "HE_DOY": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "MA_DOY": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "valid_pheno_mask": {"zlib": True, "complevel": 4, "dtype": "int8"},
    }

    ds_out.attrs["title"] = f"{crop} SEDI {scale}-month mean during two phenology stages"
    ds_out.attrs["crop"] = crop
    ds_out.attrs["sedi_scale_month"] = scale
    ds_out.attrs["time_range"] = f"{start_date} ~ {end_date}"
    ds_out.attrs["stage1_definition"] = "GR&EM <= DOY < HE"
    ds_out.attrs["stage2_definition"] = "HE <= DOY <= MA"
    ds_out.attrs["input_sedi"] = sedi_path
    ds_out.attrs["input_pheno_gr_em"] = pheno_paths["GR_EM"]
    ds_out.attrs["input_pheno_he"] = pheno_paths["HE"]
    ds_out.attrs["input_pheno_ma"] = pheno_paths["MA"]

    ds_out.to_netcdf(output_nc, encoding=encoding)
    print(f"\n✅ 结果已保存: {output_nc}")


# =========================================================
# 9. 随机绘图检查
# =========================================================
def quick_plot_check(ds_out, output_dir, crop, scale):
    years = ds_out["year"].values
    if len(years) == 0:
        print("⚠️ 没有年份可用于绘图检查。")
        return

    np.random.seed(42)
    year_pick = int(np.random.choice(years))

    s1 = ds_out["SEDI_stage1_mean"].sel(year=year_pick)
    s2 = ds_out["SEDI_stage2_mean"].sel(year=year_pick)

    fig1 = os.path.join(output_dir, f"{crop}_SEDI_{scale}month_stage1_{year_pick}.png")
    fig2 = os.path.join(output_dir, f"{crop}_SEDI_{scale}month_stage2_{year_pick}.png")

    plt.figure(figsize=(10, 5))
    s1.plot(cmap="RdBu", robust=True)
    plt.title(f"{crop} SEDI-{scale}month Stage1 Mean ({year_pick})")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.show()

    plt.figure(figsize=(10, 5))
    s2.plot(cmap="RdBu", robust=True)
    plt.title(f"{crop} SEDI-{scale}month Stage2 Mean ({year_pick})")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.show()

    print(f"\n🖼️ 随机抽检年份: {year_pick}")
    print(f"阶段1图像: {fig1}")
    print(f"阶段2图像: {fig2}")


# =========================================================
# 10. 打印结果摘要
# =========================================================
def print_output_summary(ds_out, crop, scale):
    print("\n" + "=" * 100)
    print(f"📊 输出结果摘要 | {crop} | SEDI-{scale}month")
    print("=" * 100)
    print(ds_out)

    if "year" in ds_out.dims and len(ds_out["year"]) > 0:
        y0 = int(ds_out["year"].values[0])

        s1_mean = ds_out["SEDI_stage1_mean"].sel(year=y0).mean(skipna=True).values
        s2_mean = ds_out["SEDI_stage2_mean"].sel(year=y0).mean(skipna=True).values
        s1_days = ds_out["SEDI_stage1_days"].sel(year=y0).mean(skipna=True).values
        s2_days = ds_out["SEDI_stage2_days"].sel(year=y0).mean(skipna=True).values

        print(f"\n首年: {y0}")
        print(f"首年阶段1全区平均 SEDI : {float(s1_mean):.4f}")
        print(f"首年阶段2全区平均 SEDI : {float(s2_mean):.4f}")
        print(f"首年阶段1平均有效天数  : {float(s1_days):.2f}")
        print(f"首年阶段2平均有效天数  : {float(s2_days):.2f}")


# =========================================================
# 11. 单任务执行函数
# =========================================================
def run_single_task(crop, scale):
    print("\n" + "#" * 120)
    print(f"🚀 开始任务: {crop} | SEDI-{scale}month")
    print("#" * 120)

    sedi_path = os.path.join(sedi_base_dir, f"SEDI_{scale}month_{crop}.nc")
    if not os.path.exists(sedi_path):
        print(f"❌ SEDI 文件不存在，跳过: {sedi_path}")
        return

    # 读取 SEDI
    _, sedi_da, _ = prepare_sedi_dataset(sedi_path)

    # 读取物候
    gr_da, he_da, ma_da, pheno_paths = load_pheno_for_crop(
        crop,
        target_lat=sedi_da.lat,
        target_lon=sedi_da.lon
    )

    # 检查物候
    check_pheno_validity(crop, gr_da, he_da, ma_da)

    # 计算
    ds_out = calc_stage_mean_by_year(sedi_da, gr_da, he_da, ma_da)

    # 输出路径
    out_dir = os.path.join(output_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)

    output_nc = os.path.join(out_dir, f"SEDI_{scale}month_{crop}_PhenoStageMean_2002_2022.nc")

    # 保存
    save_output_nc(ds_out, output_nc, crop, scale, sedi_path, pheno_paths)

    # 摘要
    print_output_summary(ds_out, crop, scale)

    # 随机绘图检查
    quick_plot_check(ds_out, out_dir, crop, scale)

    print(f"\n✅ 任务完成: {crop} | SEDI-{scale}month")


# =========================================================
# 12. 主程序：批量运行
# =========================================================
if __name__ == "__main__":
    print("=" * 120)
    print("开始批量计算春冬小麦各尺度 SEDI 的两个物候阶段平均值（2002-2022）")
    print("=" * 120)

    for crop in crops:
        for scale in scales:
            try:
                run_single_task(crop, scale)
            except Exception as e:
                print(f"\n❌ 任务失败: {crop} | SEDI-{scale}month")
                print(f"错误信息: {e}")
                continue

    print("\n" + "=" * 120)
    print("🎉 全部任务执行结束")
    print("=" * 120)