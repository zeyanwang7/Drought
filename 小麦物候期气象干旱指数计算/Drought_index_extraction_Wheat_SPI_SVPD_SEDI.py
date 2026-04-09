# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.features import geometry_mask
from rasterio.transform import from_origin

warnings.filterwarnings("ignore")

# =========================================================
# 1. 基本路径设置
# =========================================================
shp_path = r"E:\数据集\中国_省\中国_省2.shp"

# 输出目录
output_dir = r"E:\data\Province_Stage_Index_Mean"
os.makedirs(output_dir, exist_ok=True)

# 是否排除港澳台
exclude_provinces = ["台湾省", "香港特别行政区", "澳门特别行政区"]

# 是否绘图检查
enable_plot_check = True

# matplotlib 中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 2. 输入文件配置
#    每个指数单独一个配置字典
# =========================================================
INDEX_CONFIG = {
    "SPI": {
        "output_excel": os.path.join(output_dir, "各省SPI两阶段平均值_2002_2022.xlsx"),
        "stage1_var": "SPI_stage1_mean",
        "stage2_var": "SPI_stage2_mean",
        "files": {
            "Spring_Wheat_1month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SPI_1month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_3month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SPI_3month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_6month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SPI_6month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_9month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SPI_9month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_12month": r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SPI_12month_Spring_Wheat_PhenoStageMean_2002_2022.nc",

            "Winter_Wheat_1month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SPI_1month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_3month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SPI_3month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_6month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SPI_6month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_9month":  r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SPI_9month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_12month": r"E:\data\SPI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SPI_12month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
        }
    },

    "SEDI": {
        "output_excel": os.path.join(output_dir, "各省SEDI两阶段平均值_2002_2022.xlsx"),
        "stage1_var": "SEDI_stage1_mean",
        "stage2_var": "SEDI_stage2_mean",
        "files": {
            "Spring_Wheat_1month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SEDI_1month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_3month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SEDI_3month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_6month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SEDI_6month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_9month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SEDI_9month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_12month": r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Spring_Wheat\SEDI_12month_Spring_Wheat_PhenoStageMean_2002_2022.nc",

            "Winter_Wheat_1month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SEDI_1month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_3month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SEDI_3month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_6month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SEDI_6month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_9month":  r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SEDI_9month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_12month": r"E:\data\SEDI\Wheat\Pheno_Stage_Mean\Winter_Wheat\SEDI_12month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
        }
    },

    "SVPD": {
        "output_excel": os.path.join(output_dir, "各省SVPD两阶段平均值_2002_2022.xlsx"),
        "stage1_var": "SVPD_stage1_mean",
        "stage2_var": "SVPD_stage2_mean",
        "files": {
            "Spring_Wheat_1month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Spring_Wheat\SVPD_1month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_3month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Spring_Wheat\SVPD_3month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_6month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Spring_Wheat\SVPD_6month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_9month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Spring_Wheat\SVPD_9month_Spring_Wheat_PhenoStageMean_2002_2022.nc",
            "Spring_Wheat_12month": r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Spring_Wheat\SVPD_12month_Spring_Wheat_PhenoStageMean_2002_2022.nc",

            "Winter_Wheat_1month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Winter_Wheat\SVPD_1month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_3month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Winter_Wheat\SVPD_3month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_6month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Winter_Wheat\SVPD_6month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_9month":  r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Winter_Wheat\SVPD_9month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
            "Winter_Wheat_12month": r"E:\data\SVPD\Wheat\Pheno_Stage_Mean\Winter_Wheat\SVPD_12month_Winter_Wheat_PhenoStageMean_2002_2022.nc",
        }
    }
}


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

    return from_origin(west, north, lon_res, lat_res)


def ensure_lat_desc(da):
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.sortby("lat", ascending=False)
    return da


def open_stage_nc(nc_path, stage1_var, stage2_var):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"NC 文件不存在: {nc_path}")

    ds = xr.open_dataset(nc_path)

    for v in [stage1_var, stage2_var]:
        if v not in ds.data_vars:
            raise ValueError(f"{nc_path} 中缺少变量: {v}")

    if not {"year", "lat", "lon"}.issubset(set(ds[stage1_var].dims)):
        raise ValueError(
            f"{nc_path} 中 {stage1_var} 维度不是(year, lat, lon)，"
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

    return ds, stage1, stage2, valid_mask, years, lat_vals, lon_vals


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

    # 删除阶段1和阶段2都为空的记录
    df = df.dropna(
        subset=[f"阶段1平均{index_name}", f"阶段2平均{index_name}"],
        how="all"
    ).reset_index(drop=True)

    return df


def quick_plot_check(df, task_name, index_name):
    if len(df) == 0:
        return

    year_pick = int(np.random.choice(df["年份"].unique()))
    sub = df[df["年份"] == year_pick].copy()

    if len(sub) == 0:
        return

    print("\n" + "-" * 100)
    print(f"🖼️ 绘图检查: {task_name} | 年份 = {year_pick}")
    print("-" * 100)

    c1 = f"阶段1平均{index_name}"
    c2 = f"阶段2平均{index_name}"

    sub1 = sub.dropna(subset=[c1]).sort_values(c1)
    if len(sub1) > 0:
        plt.figure(figsize=(14, 6))
        plt.bar(sub1["省份"], sub1[c1])
        plt.xticks(rotation=90)
        plt.title(f"{task_name} - {year_pick}年 阶段1省均值")
        plt.ylabel(c1)
        plt.tight_layout()
        plt.show()

    sub2 = sub.dropna(subset=[c2]).sort_values(c2)
    if len(sub2) > 0:
        plt.figure(figsize=(14, 6))
        plt.bar(sub2["省份"], sub2[c2])
        plt.xticks(rotation=90)
        plt.title(f"{task_name} - {year_pick}年 阶段2省均值")
        plt.ylabel(c2)
        plt.tight_layout()
        plt.show()


def process_one_index(index_name, config, gdf):
    output_excel = config["output_excel"]
    stage1_var = config["stage1_var"]
    stage2_var = config["stage2_var"]
    files = config["files"]

    print("\n" + "=" * 120)
    print(f"开始处理指数: {index_name}")
    print("=" * 120)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        all_results = []

        for task_name, nc_path in files.items():
            print("\n" + "=" * 120)
            print(f"处理任务: {task_name}")
            print(f"文件路径: {nc_path}")
            print("=" * 120)

            if not os.path.exists(nc_path):
                print(f"⚠️ 文件不存在，跳过: {nc_path}")
                continue

            ds, stage1, stage2, valid_mask, years, lat_vals, lon_vals = open_stage_nc(
                nc_path, stage1_var, stage2_var
            )

            print(f"年份范围: {years.min()} - {years.max()} | 年数: {len(years)}")
            print(f"网格大小: lat={len(lat_vals)}, lon={len(lon_vals)}")
            print(f"纬度方向: {'降序' if lat_vals[0] > lat_vals[-1] else '升序'}")

            if valid_mask is not None:
                print("✅ 检测到 valid_pheno_mask，将仅在有效物候区内统计")
            else:
                print("⚠️ 未检测到 valid_pheno_mask，将直接对阶段均值图层统计")

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
            if len(parts) >= 3:
                crop = "_".join(parts[:2])
                scale = parts[2]
            else:
                crop = task_name
                scale = ""

            df_result.insert(0, "任务名", task_name)
            df_result.insert(1, "作物", crop)
            df_result.insert(2, "尺度", scale)

            if len(df_result) > 0:
                sheet_name = task_name[:31]
                df_result.to_excel(writer, sheet_name=sheet_name, index=False)
                all_results.append(df_result)

                if enable_plot_check:
                    quick_plot_check(df_result, task_name, index_name)
            else:
                print(f"⚠️ {task_name} 结果全为空，跳过写入 Excel")

            ds.close()

        if len(all_results) > 0:
            df_all = pd.concat(all_results, ignore_index=True)
            df_all = df_all.dropna(
                subset=[f"阶段1平均{index_name}", f"阶段2平均{index_name}"],
                how="all"
            ).reset_index(drop=True)
            df_all.to_excel(writer, sheet_name="All_Results", index=False)

    print(f"\n🎉 {index_name} 全部完成")
    print(f"结果已保存到: {output_excel}")


# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("=" * 120)
    print("开始计算冬春小麦 SPI / SEDI / SVPD 各省两阶段平均值")
    print("=" * 120)

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"shp 文件不存在: {shp_path}")

    # 显式指定 UTF-8
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
    print("示例省份:", gdf["province_name"].head().tolist())

    for index_name, config in INDEX_CONFIG.items():
        process_one_index(index_name, config, gdf)

    print("\n" + "=" * 120)
    print("🎉 全部指数处理结束")
    print("=" * 120)


if __name__ == "__main__":
    main()