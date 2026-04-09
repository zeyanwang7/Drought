import os
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt

# =========================================================
# 1. 基本参数设置
# =========================================================
pheno_base_dir = r"E:\数据集\重采样后物候期\玉米\0.1度_修复版"
output_base_root = r"E:\data\Maize_Pheno_Stage_Mean"

os.makedirs(output_base_root, exist_ok=True)

crops = ["Spring_Maize", "Summer_Maize"]
scales = ["1month", "3month", "6month", "9month", "12month"]

start_date = "2002-01-01"
end_date   = "2022-12-31"

# =========================================================
# 2. 指数配置
#    这里按你之前四个指数代码中的路径和文件名规则来设置
# =========================================================
index_configs = {
    # "SPET": {
    #     "input_dir": r"E:\data\SPET\Maize",
    #     "file_pattern": "SPET_{scale}_{crop}.nc",
    #     "output_dir": os.path.join(output_base_root, "SPET"),
    #     "preferred_vars": ["SPET"]
    # },
    # "SPEI": {
    #     "input_dir": r"E:\data\SPEI\Maize",
    #     "file_pattern": "SPEI_{scale}_{crop}.nc",
    #     "output_dir": os.path.join(output_base_root, "SPEI"),
    #     "preferred_vars": ["SPEI"]
    # },
    "SVPD": {
        "input_dir": r"E:\data\SVPD\Maize_fast",
        "file_pattern": "SVPD_{scale}_{crop}.nc",
        "output_dir": os.path.join(output_base_root, "SVPD"),
        "preferred_vars": ["SVPD"]
    }
#     "SEDI": {
#         "input_dir": r"E:\data\SEDI\Maize",
#         "file_pattern": "SEDI_{scale}_{crop}.nc",
#         "output_dir": os.path.join(output_base_root, "SEDI"),
#         "preferred_vars": ["SEDI"]
#     }
}

for cfg in index_configs.values():
    os.makedirs(cfg["output_dir"], exist_ok=True)


# =========================================================
# 3. 自动识别变量名
# =========================================================
def detect_main_var(ds, preferred_vars=None):
    if preferred_vars is None:
        preferred_vars = []

    for v in preferred_vars:
        if v in ds.data_vars:
            return v

    # 再按名称模糊匹配
    lower_vars = {k.lower(): k for k in ds.data_vars}
    for v in preferred_vars:
        if v.lower() in lower_vars:
            return lower_vars[v.lower()]

    # 如果只有一个变量，直接用
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    # 兜底：优先找三维变量
    for v in ds.data_vars:
        if ds[v].ndim in [2, 3]:
            return v

    raise ValueError(f"无法自动识别主变量，当前变量有: {list(ds.data_vars)}")


# =========================================================
# 4. 读取并标准化指数数据
# =========================================================
def prepare_index_dataset(index_path, preferred_vars=None):
    ds = xr.open_dataset(index_path)

    print("=" * 100)
    print(f"📂 读取指数文件: {index_path}")
    print("=" * 100)
    print(ds)

    var_name = detect_main_var(ds, preferred_vars=preferred_vars)
    print(f"\n✅ 自动识别变量名: {var_name}")

    da = ds[var_name]

    # squeeze 掉长度为1的多余维度
    da = da.squeeze(drop=True)

    # 维度检查
    if "time" not in da.dims:
        raise ValueError(f"变量中缺少 time 维，当前维度为: {da.dims}")

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"变量中缺少 lat/lon 维，当前维度为: {da.dims}")

    da = da.transpose("time", "lat", "lon")

    # 纬度升序
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")

    # 时间截取
    da = da.sel(time=slice(start_date, end_date))

    print(f"\n📌 时间范围: {str(da.time.values[0])[:10]} ~ {str(da.time.values[-1])[:10]}")
    print(f"📌 数据 shape: {da.shape}")

    return ds, da, var_name


# =========================================================
# 5. 读取 tif 并转为与指数网格对齐的 DataArray
# =========================================================
def read_tif_as_da(tif_path, target_lat=None, target_lon=None, name="pheno"):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        transform = src.transform
        height = src.height
        width = src.width

        if nodata is not None:
            arr[arr == nodata] = np.nan

        for invalid in [-9999, -999, 0]:
            arr[arr == invalid] = np.nan

        x_coords = np.array(
            [transform.c + transform.a * (i + 0.5) for i in range(width)],
            dtype=np.float64
        )
        y_coords = np.array(
            [transform.f + transform.e * (j + 0.5) for j in range(height)],
            dtype=np.float64
        )

        # 若纬度降序，则翻转
        if y_coords[0] > y_coords[-1]:
            y_coords = y_coords[::-1]
            arr = np.flipud(arr)

        da = xr.DataArray(
            arr,
            coords={"lat": y_coords, "lon": x_coords},
            dims=("lat", "lon"),
            name=name
        )

        if target_lat is not None and target_lon is not None:
            da = da.interp(lat=target_lat, lon=target_lon, method="nearest")

    return da, profile


# =========================================================
# 6. 读取玉米物候文件
# =========================================================
def get_pheno_paths(crop):
    return {
        "V3": os.path.join(pheno_base_dir, f"{crop}_V3_0.1deg_fixed.tif"),
        "HE": os.path.join(pheno_base_dir, f"{crop}_HE_0.1deg_fixed.tif"),
        "MA": os.path.join(pheno_base_dir, f"{crop}_MA_0.1deg_fixed.tif"),
    }


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

    v3_da, _ = read_tif_as_da(paths["V3"], target_lat=target_lat, target_lon=target_lon, name="V3_DOY")
    he_da, _ = read_tif_as_da(paths["HE"], target_lat=target_lat, target_lon=target_lon, name="HE_DOY")
    ma_da, _ = read_tif_as_da(paths["MA"], target_lat=target_lat, target_lon=target_lon, name="MA_DOY")

    print(f"✅ {crop} 物候数据读取完成并已对齐到指数网格")
    print(f"V3 shape: {v3_da.shape}")
    print(f"HE shape: {he_da.shape}")
    print(f"MA shape: {ma_da.shape}")

    return v3_da, he_da, ma_da, paths


# =========================================================
# 7. 检查物候有效性
# =========================================================
def check_pheno_validity(crop, v3_da, he_da, ma_da):
    valid_all = np.isfinite(v3_da) & np.isfinite(he_da) & np.isfinite(ma_da)
    valid_triplet = valid_all & (v3_da < he_da) & (he_da < ma_da)

    valid_count = int(valid_all.sum().values)
    ok_count = int(valid_triplet.sum().values)

    print("\n" + "-" * 100)
    print(f"🔍 {crop} 物候有效性检查")
    print("-" * 100)
    print(f"三期同时有效像元数            : {valid_count}")
    print(f"满足 V3 < HE < MA 的像元数    : {ok_count}")
    print(f"不满足顺序的像元数            : {valid_count - ok_count}")

    if ok_count > 0:
        s1 = (he_da - v3_da).where(valid_triplet)
        s2 = (ma_da - he_da).where(valid_triplet)
        total = (ma_da - v3_da).where(valid_triplet)

        print(f"阶段1长度均值(HE - V3): {float(s1.mean(skipna=True).values):.2f} 天")
        print(f"阶段2长度均值(MA - HE): {float(s2.mean(skipna=True).values):.2f} 天")
        print(f"总长度均值(MA - V3)  : {float(total.mean(skipna=True).values):.2f} 天")


# =========================================================
# 8. 核心函数：计算逐年两个阶段平均值
# =========================================================
def calc_stage_mean_by_year(index_da, v3_da, he_da, ma_da, index_name):
    valid_pheno = (
        np.isfinite(v3_da) & np.isfinite(he_da) & np.isfinite(ma_da) &
        (v3_da < he_da) & (he_da < ma_da)
    )

    # 四舍五入为整数 DOY
    v3_da = xr.where(valid_pheno, np.rint(v3_da), np.nan)
    he_da = xr.where(valid_pheno, np.rint(he_da), np.nan)
    ma_da = xr.where(valid_pheno, np.rint(ma_da), np.nan)

    # 每日 DOY
    doy = xr.DataArray(
        index_da.time.dt.dayofyear.values,
        coords={"time": index_da.time},
        dims=("time",),
        name="doy"
    )

    # 广播
    doy_3d, v3_3d = xr.broadcast(doy, v3_da)
    _, he_3d = xr.broadcast(doy, he_da)
    _, ma_3d = xr.broadcast(doy, ma_da)
    _, valid_3d = xr.broadcast(doy, valid_pheno)

    # 两阶段掩膜
    stage1_mask = valid_3d & (doy_3d >= v3_3d) & (doy_3d < he_3d)
    stage2_mask = valid_3d & (doy_3d >= he_3d) & (doy_3d <= ma_3d)

    # 阶段内指数
    idx_stage1 = index_da.where(stage1_mask)
    idx_stage2 = index_da.where(stage2_mask)

    # 按年求平均
    stage1_mean = idx_stage1.groupby("time.year").mean(dim="time", skipna=True)
    stage2_mean = idx_stage2.groupby("time.year").mean(dim="time", skipna=True)

    # 有效天数
    stage1_days = (stage1_mask & np.isfinite(index_da)).groupby("time.year").sum(dim="time")
    stage2_days = (stage2_mask & np.isfinite(index_da)).groupby("time.year").sum(dim="time")

    # 变量名
    stage1_mean = stage1_mean.rename(f"{index_name}_stage1_mean").astype(np.float32)
    stage2_mean = stage2_mean.rename(f"{index_name}_stage2_mean").astype(np.float32)
    stage1_days = stage1_days.rename(f"{index_name}_stage1_days").astype(np.int16)
    stage2_days = stage2_days.rename(f"{index_name}_stage2_days").astype(np.int16)

    v3_da = v3_da.rename("V3_DOY").astype(np.float32)
    he_da = he_da.rename("HE_DOY").astype(np.float32)
    ma_da = ma_da.rename("MA_DOY").astype(np.float32)
    valid_pheno = valid_pheno.rename("valid_pheno_mask").astype(np.int8)

    out_ds = xr.Dataset({
        f"{index_name}_stage1_mean": stage1_mean,
        f"{index_name}_stage2_mean": stage2_mean,
        f"{index_name}_stage1_days": stage1_days,
        f"{index_name}_stage2_days": stage2_days,
        "V3_DOY": v3_da,
        "HE_DOY": he_da,
        "MA_DOY": ma_da,
        "valid_pheno_mask": valid_pheno
    })

    return out_ds


# =========================================================
# 9. 保存 nc
# =========================================================
def save_output_nc(ds_out, output_nc, crop, scale, index_name, input_path, pheno_paths):
    encoding = {}

    for v in ds_out.data_vars:
        if v.endswith("_mean"):
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif v.endswith("_days"):
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "int16"}
        elif v in ["V3_DOY", "HE_DOY", "MA_DOY"]:
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif v == "valid_pheno_mask":
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "int8"}

    ds_out.attrs["title"] = f"{crop} {index_name} {scale} mean during two phenology stages"
    ds_out.attrs["crop"] = crop
    ds_out.attrs["index_name"] = index_name
    ds_out.attrs["time_scale"] = scale
    ds_out.attrs["time_range"] = f"{start_date} ~ {end_date}"
    ds_out.attrs["stage1_definition"] = "V3 <= DOY < HE"
    ds_out.attrs["stage2_definition"] = "HE <= DOY <= MA"
    ds_out.attrs["input_index"] = input_path
    ds_out.attrs["input_pheno_v3"] = pheno_paths["V3"]
    ds_out.attrs["input_pheno_he"] = pheno_paths["HE"]
    ds_out.attrs["input_pheno_ma"] = pheno_paths["MA"]

    ds_out.to_netcdf(output_nc, encoding=encoding)
    print(f"\n✅ 结果已保存: {output_nc}")


# =========================================================
# 10. 随机绘图检查
# =========================================================
def quick_plot_check(ds_out, output_dir, crop, scale, index_name):
    years = ds_out["year"].values
    if len(years) == 0:
        print("⚠️ 没有年份可用于绘图检查。")
        return

    np.random.seed(42)
    year_pick = int(np.random.choice(years))

    s1 = ds_out[f"{index_name}_stage1_mean"].sel(year=year_pick)
    s2 = ds_out[f"{index_name}_stage2_mean"].sel(year=year_pick)

    fig1 = os.path.join(output_dir, f"{crop}_{index_name}_{scale}_stage1_{year_pick}.png")
    fig2 = os.path.join(output_dir, f"{crop}_{index_name}_{scale}_stage2_{year_pick}.png")

    plt.figure(figsize=(10, 5))
    s1.plot(cmap="RdBu", robust=True)
    plt.title(f"{crop} {index_name}-{scale} Stage1 Mean ({year_pick})")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.show()

    plt.figure(figsize=(10, 5))
    s2.plot(cmap="RdBu", robust=True)
    plt.title(f"{crop} {index_name}-{scale} Stage2 Mean ({year_pick})")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.show()

    print(f"\n🖼️ 随机抽检年份: {year_pick}")
    print(f"阶段1图像: {fig1}")
    print(f"阶段2图像: {fig2}")


# =========================================================
# 11. 打印摘要
# =========================================================
def print_output_summary(ds_out, crop, scale, index_name):
    print("\n" + "=" * 100)
    print(f"📊 输出结果摘要 | {crop} | {index_name}-{scale}")
    print("=" * 100)
    print(ds_out)

    if "year" in ds_out.dims and len(ds_out["year"]) > 0:
        y0 = int(ds_out["year"].values[0])
        print(f"\n首年: {y0}")

        s1_mean = ds_out[f"{index_name}_stage1_mean"].sel(year=y0).mean(skipna=True).values
        s2_mean = ds_out[f"{index_name}_stage2_mean"].sel(year=y0).mean(skipna=True).values
        s1_days = ds_out[f"{index_name}_stage1_days"].sel(year=y0).mean(skipna=True).values
        s2_days = ds_out[f"{index_name}_stage2_days"].sel(year=y0).mean(skipna=True).values

        print(f"首年阶段1全区平均 {index_name} : {float(s1_mean):.4f}")
        print(f"首年阶段2全区平均 {index_name} : {float(s2_mean):.4f}")
        print(f"首年阶段1平均有效天数       : {float(s1_days):.2f}")
        print(f"首年阶段2平均有效天数       : {float(s2_days):.2f}")


# =========================================================
# 12. 单任务执行
# =========================================================
def run_single_task(index_name, crop, scale):
    print("\n" + "#" * 120)
    print(f"🚀 开始任务: {index_name} | {crop} | {scale}")
    print("#" * 120)

    cfg = index_configs[index_name]
    input_path = os.path.join(cfg["input_dir"], cfg["file_pattern"].format(scale=scale, crop=crop))

    if not os.path.exists(input_path):
        print(f"❌ 指数文件不存在，跳过: {input_path}")
        return

    # 读取指数
    _, index_da, _ = prepare_index_dataset(input_path, preferred_vars=cfg["preferred_vars"])

    # 读取物候
    v3_da, he_da, ma_da, pheno_paths = load_pheno_for_crop(
        crop,
        target_lat=index_da.lat,
        target_lon=index_da.lon
    )

    # 检查物候
    check_pheno_validity(crop, v3_da, he_da, ma_da)

    # 计算
    ds_out = calc_stage_mean_by_year(index_da, v3_da, he_da, ma_da, index_name=index_name)

    # 输出路径
    out_dir = os.path.join(cfg["output_dir"], crop)
    os.makedirs(out_dir, exist_ok=True)

    output_nc = os.path.join(out_dir, f"{index_name}_{scale}_{crop}_PhenoStageMean_2002_2022.nc")

    # 保存
    save_output_nc(ds_out, output_nc, crop, scale, index_name, input_path, pheno_paths)

    # 摘要
    print_output_summary(ds_out, crop, scale, index_name)

    # 绘图检查
    quick_plot_check(ds_out, out_dir, crop, scale, index_name)

    print(f"\n✅ 任务完成: {index_name} | {crop} | {scale}")


# =========================================================
# 13. 主程序：批量运行四类指数
# =========================================================
if __name__ == "__main__":
    print("=" * 120)
    print("开始批量计算春玉米 / 夏玉米 两阶段平均干旱指数（SPET / SPEI / SVPD / SEDI）")
    print("=" * 120)

    for index_name in ["SPET", "SPEI", "SVPD", "SEDI"]:
        for crop in crops:
            for scale in scales:
                try:
                    run_single_task(index_name, crop, scale)
                except Exception as e:
                    print(f"\n❌ 任务失败: {index_name} | {crop} | {scale}")
                    print(f"错误信息: {e}")
                    continue

    print("\n" + "=" * 120)
    print("🎉 全部任务执行结束")
    print("=" * 120)