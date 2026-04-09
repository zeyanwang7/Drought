import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from rasterio.features import geometry_mask
from rasterio.transform import from_origin

warnings.filterwarnings("ignore")

# =========================================================
# 1. 路径设置
# =========================================================
shp_path = r"E:\数据集\中国_省\中国_省2.shp"
output_excel = r"E:\data\Province_Stage_Index_Mean\Maize_各省两阶段平均干旱指数_2002_2022.xlsx"

# =========================================================
# 2. 这里配置你要提取的玉米两阶段均值 NC 文件
#    你可以按需删减，只保留你已经算好的指数
# =========================================================
input_nc_files = {
    # ===================== SPI =====================
    "Spring_Maize_SPI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Spring_Maize\SPI_1month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Spring_Maize\SPI_3month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Spring_Maize\SPI_6month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Spring_Maize\SPI_9month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPI\Spring_Maize\SPI_12month_Spring_Maize_PhenoStageMean_2002_2022.nc",

    "Summer_Maize_SPI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Summer_Maize\SPI_1month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Summer_Maize\SPI_3month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Summer_Maize\SPI_6month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPI\Summer_Maize\SPI_9month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPI\Summer_Maize\SPI_12month_Summer_Maize_PhenoStageMean_2002_2022.nc",

    # ===================== SPET =====================
    "Spring_Maize_SPET_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Spring_Maize\SPET_1month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPET_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Spring_Maize\SPET_3month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPET_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Spring_Maize\SPET_6month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPET_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Spring_Maize\SPET_9month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPET_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPET\Spring_Maize\SPET_12month_Spring_Maize_PhenoStageMean_2002_2022.nc",

    "Summer_Maize_SPET_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Summer_Maize\SPET_1month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPET_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Summer_Maize\SPET_3month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPET_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Summer_Maize\SPET_6month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPET_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPET\Summer_Maize\SPET_9month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPET_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPET\Summer_Maize\SPET_12month_Summer_Maize_PhenoStageMean_2002_2022.nc",

    # ===================== SPEI =====================
    "Spring_Maize_SPEI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Spring_Maize\SPEI_1month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPEI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Spring_Maize\SPEI_3month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPEI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Spring_Maize\SPEI_6month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPEI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Spring_Maize\SPEI_9month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SPEI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Spring_Maize\SPEI_12month_Spring_Maize_PhenoStageMean_2002_2022.nc",

    "Summer_Maize_SPEI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Summer_Maize\SPEI_1month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPEI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Summer_Maize\SPEI_3month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPEI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Summer_Maize\SPEI_6month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPEI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Summer_Maize\SPEI_9month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SPEI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SPEI\Summer_Maize\SPEI_12month_Summer_Maize_PhenoStageMean_2002_2022.nc",

    # ===================== SVPD =====================
    "Spring_Maize_SVPD_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Spring_Maize\SVPD_1month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SVPD_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Spring_Maize\SVPD_3month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SVPD_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Spring_Maize\SVPD_6month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SVPD_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Spring_Maize\SVPD_9month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SVPD_12month": r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Spring_Maize\SVPD_12month_Spring_Maize_PhenoStageMean_2002_2022.nc",

    "Summer_Maize_SVPD_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Summer_Maize\SVPD_1month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SVPD_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Summer_Maize\SVPD_3month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SVPD_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Summer_Maize\SVPD_6month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SVPD_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Summer_Maize\SVPD_9month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SVPD_12month": r"E:\data\Maize_Pheno_Stage_Mean\SVPD\Summer_Maize\SVPD_12month_Summer_Maize_PhenoStageMean_2002_2022.nc",

    # ===================== SEDI =====================
    "Spring_Maize_SEDI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Spring_Maize\SEDI_1month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SEDI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Spring_Maize\SEDI_3month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SEDI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Spring_Maize\SEDI_6month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SEDI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Spring_Maize\SEDI_9month_Spring_Maize_PhenoStageMean_2002_2022.nc",
    "Spring_Maize_SEDI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Spring_Maize\SEDI_12month_Spring_Maize_PhenoStageMean_2002_2022.nc",

    "Summer_Maize_SEDI_1month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Summer_Maize\SEDI_1month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SEDI_3month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Summer_Maize\SEDI_3month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SEDI_6month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Summer_Maize\SEDI_6month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SEDI_9month":  r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Summer_Maize\SEDI_9month_Summer_Maize_PhenoStageMean_2002_2022.nc",
    "Summer_Maize_SEDI_12month": r"E:\data\Maize_Pheno_Stage_Mean\SEDI\Summer_Maize\SEDI_12month_Summer_Maize_PhenoStageMean_2002_2022.nc",
}

exclude_provinces = ["台湾省", "香港特别行政区", "澳门特别行政区"]


# =========================================================
# 3. 工具函数
# =========================================================
def get_province_name_column(gdf):
    candidates = ["省", "省份", "NAME", "name", "NAME_CHN", "NAME_1", "province", "Prov_Name"]
    for c in candidates:
        if c in gdf.columns:
            return c

    non_geo_cols = [c for c in gdf.columns if c != "geometry"]
    if len(non_geo_cols) == 0:
        raise ValueError("未找到省份名称字段，请手动指定。")
    return non_geo_cols[0]


def build_transform_from_latlon(lat_vals, lon_vals):
    lat_vals = np.asarray(lat_vals)
    lon_vals = np.asarray(lon_vals)

    lat_res = abs(lat_vals[1] - lat_vals[0])
    lon_res = abs(lon_vals[1] - lon_vals[0])

    west = lon_vals.min() - lon_res / 2
    north = lat_vals.max() + lat_res / 2

    transform = from_origin(west, north, lon_res, lat_res)
    return transform


def ensure_lat_desc(da):
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.sortby("lat", ascending=False)
    return da


def detect_stage_var_names(ds):
    data_vars = list(ds.data_vars)

    stage1_candidates = [v for v in data_vars if v.lower().endswith("stage1_mean")]
    stage2_candidates = [v for v in data_vars if v.lower().endswith("stage2_mean")]

    if len(stage1_candidates) == 0 or len(stage2_candidates) == 0:
        raise ValueError(f"未识别到 stage1/stage2 均值变量，当前变量有: {data_vars}")

    if len(stage1_candidates) == 1 and len(stage2_candidates) == 1:
        stage1_var = stage1_candidates[0]
        stage2_var = stage2_candidates[0]
    else:
        stage1_var = None
        stage2_var = None
        for v1 in stage1_candidates:
            prefix1 = v1.lower().replace("_stage1_mean", "")
            for v2 in stage2_candidates:
                prefix2 = v2.lower().replace("_stage2_mean", "")
                if prefix1 == prefix2:
                    stage1_var = v1
                    stage2_var = v2
                    break
            if stage1_var is not None:
                break

        if stage1_var is None or stage2_var is None:
            raise ValueError(f"无法配对 stage1/stage2 变量，当前变量有: {data_vars}")

    index_name = stage1_var.replace("_stage1_mean", "").replace("_stage1_mean".upper(), "")
    return stage1_var, stage2_var, index_name


def open_stage_nc(nc_path):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"NC 文件不存在: {nc_path}")

    ds = xr.open_dataset(nc_path)

    stage1_var, stage2_var, index_name = detect_stage_var_names(ds)

    if not {"year", "lat", "lon"}.issubset(set(ds[stage1_var].dims)):
        raise ValueError(
            f"{nc_path} 中 {stage1_var} 维度不是(year, lat, lon)体系，"
            f"当前为: {ds[stage1_var].dims}"
        )

    stage1 = ds[stage1_var].transpose("year", "lat", "lon").astype(np.float32)
    stage2 = ds[stage2_var].transpose("year", "lat", "lon").astype(np.float32)

    stage1 = ensure_lat_desc(stage1)
    stage2 = ensure_lat_desc(stage2)

    valid_mask = None
    if "valid_pheno_mask" in ds.data_vars:
        valid_mask = ds["valid_pheno_mask"]
        if set(valid_mask.dims) == {"lat", "lon"}:
            valid_mask = valid_mask.transpose("lat", "lon")
            if valid_mask["lat"].values[0] < valid_mask["lat"].values[-1]:
                valid_mask = valid_mask.sortby("lat", ascending=False)
            valid_mask = valid_mask.astype(np.int8)

    years = ds["year"].values.astype(int)
    lat_vals = stage1["lat"].values
    lon_vals = stage1["lon"].values

    return ds, stage1, stage2, valid_mask, years, lat_vals, lon_vals, stage1_var, stage2_var, index_name


def build_province_masks(gdf, lat_vals, lon_vals):
    transform = build_transform_from_latlon(lat_vals, lon_vals)

    province_masks = {}
    for _, row in gdf.iterrows():
        province = row["province_name"]
        geom = row.geometry

        mask = geometry_mask(
            [geom],
            out_shape=(len(lat_vals), len(lon_vals)),
            transform=transform,
            invert=True,
            all_touched=True
        )
        province_masks[province] = mask

    print(f"✅ 已构建 {len(province_masks)} 个省级 mask")
    return province_masks


def aggregate_stage_mean_by_province(stage1, stage2, valid_mask, years, province_masks, index_name):
    results = []

    for i, year in enumerate(years):
        arr1 = stage1.isel(year=i).values.astype(np.float32)
        arr2 = stage2.isel(year=i).values.astype(np.float32)

        if valid_mask is not None:
            vm = (valid_mask.values == 1)
            arr1 = np.where(vm, arr1, np.nan)
            arr2 = np.where(vm, arr2, np.nan)

        for province, p_mask in province_masks.items():
            vals1 = arr1[p_mask]
            vals2 = arr2[p_mask]

            mean1 = np.nan if np.all(np.isnan(vals1)) else float(np.nanmean(vals1))
            mean2 = np.nan if np.all(np.isnan(vals2)) else float(np.nanmean(vals2))

            n1 = int(np.sum(~np.isnan(vals1)))
            n2 = int(np.sum(~np.isnan(vals2)))

            results.append({
                "年份": int(year),
                "省份": province,
                f"阶段1平均{index_name}": mean1,
                f"阶段2平均{index_name}": mean2,
                "阶段1有效像元数": n1,
                "阶段2有效像元数": n2
            })

        print(f"✅ {int(year)} 年完成")

    df = pd.DataFrame(results)

    df = df.dropna(subset=[f"阶段1平均{index_name}", f"阶段2平均{index_name}"], how="all").reset_index(drop=True)

    return df


# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("=" * 120)
    print("开始计算：春玉米 / 夏玉米 两阶段物候平均干旱指数 -> 省级平均值")
    print("=" * 120)

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"shp 文件不存在: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding="utf-8")

    province_col = get_province_name_column(gdf)

    gdf = gdf[[province_col, "geometry"]].copy()
    gdf = gdf.rename(columns={province_col: "province_name"})
    gdf = gdf[~gdf.geometry.isna()].copy()

    if exclude_provinces:
        gdf = gdf[~gdf["province_name"].isin(exclude_provinces)].copy()

    if gdf.crs is None:
        print("⚠️ shp 未检测到 CRS，默认设为 EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    print(f"✅ 省界读取完成，共 {len(gdf)} 个省级单元")
    print(f"✅ 省名字段: province_name")
    print("示例省份:", gdf["province_name"].head().tolist())

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        all_results = []

        for task_name, nc_path in input_nc_files.items():
            print("\n" + "=" * 120)
            print(f"处理任务: {task_name}")
            print(f"文件路径: {nc_path}")
            print("=" * 120)

            if not os.path.exists(nc_path):
                print(f"⚠️ 文件不存在，跳过: {nc_path}")
                continue

            ds, stage1, stage2, valid_mask, years, lat_vals, lon_vals, stage1_var, stage2_var, index_name = open_stage_nc(nc_path)

            print(f"✅ 识别到阶段变量: {stage1_var}, {stage2_var}")
            print(f"✅ 识别到指数名: {index_name}")

            if valid_mask is not None:
                print("✅ 检测到 valid_pheno_mask，将仅在有效物候区内统计")
            else:
                print("⚠️ 未检测到 valid_pheno_mask，将直接对阶段均值图层统计")

            print(f"年份范围: {years.min()} - {years.max()} | 年数: {len(years)}")
            print(f"网格大小: lat={len(lat_vals)}, lon={len(lon_vals)}")
            print(f"纬度方向: {'降序' if lat_vals[0] > lat_vals[-1] else '升序'}")

            province_masks = build_province_masks(gdf, lat_vals, lon_vals)

            df_result = aggregate_stage_mean_by_province(
                stage1=stage1,
                stage2=stage2,
                valid_mask=valid_mask,
                years=years,
                province_masks=province_masks,
                index_name=index_name
            )

            parts = task_name.split("_")
            if len(parts) >= 4:
                crop = "_".join(parts[:2])
                index_type = parts[2]
                scale = parts[3]
            else:
                crop = task_name
                index_type = ""
                scale = ""

            df_result.insert(0, "任务名", task_name)
            df_result.insert(1, "作物", crop)
            df_result.insert(2, "指数", index_type)
            df_result.insert(3, "尺度", scale)

            if len(df_result) > 0:
                sheet_name = task_name[:31]
                df_result.to_excel(writer, sheet_name=sheet_name, index=False)
                all_results.append(df_result)
            else:
                print(f"⚠️ {task_name} 结果全为空，跳过写入 Excel")

            ds.close()

        if len(all_results) > 0:
            df_all = pd.concat(all_results, ignore_index=True)
            value_cols = [c for c in df_all.columns if c.startswith("阶段1平均") or c.startswith("阶段2平均")]
            if len(value_cols) >= 2:
                df_all = df_all.dropna(subset=value_cols, how="all").reset_index(drop=True)
            df_all.to_excel(writer, sheet_name="All_Results", index=False)

    print("\n" + "=" * 120)
    print("🎉 全部完成")
    print(f"结果已保存到: {output_excel}")
    print("=" * 120)


if __name__ == "__main__":
    main()