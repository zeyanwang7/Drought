import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# =========================================================
# 1. 路径设置
# =========================================================
base_dir = r"E:\data\ET_minus_PET_Daily"
years = [2007, 2008, 2009, 2010, 2011]

# 输出结果
output_excel = os.path.join(base_dir, "ET_PET_2007_2011_统计检查.xlsx")

# =========================================================
# 2. 自动识别变量名
# =========================================================
def detect_var_name(ds):
    # 优先识别 Diff
    for v in ds.data_vars:
        if v.lower() == "diff":
            return v
    # 如果没有 Diff，就取第一个变量
    return list(ds.data_vars)[0]

# =========================================================
# 3. 读取单文件统计
# =========================================================
def calc_single_file_stats(nc_file):
    ds = xr.open_dataset(nc_file)

    var_name = detect_var_name(ds)
    da = ds[var_name]

    # 转为 numpy
    arr = da.values.astype(np.float64)

    # 若有 time 维，去掉长度为1的 time
    arr = np.squeeze(arr)

    total_pixels = arr.size
    valid_mask = np.isfinite(arr)
    valid_count = np.sum(valid_mask)

    if valid_count == 0:
        stats = {
            "file": os.path.basename(nc_file),
            "var_name": var_name,
            "shape": str(arr.shape),
            "total_pixels": total_pixels,
            "valid_pixels": 0,
            "valid_ratio_%": 0.0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan
        }
    else:
        valid_values = arr[valid_mask]
        stats = {
            "file": os.path.basename(nc_file),
            "var_name": var_name,
            "shape": str(arr.shape),
            "total_pixels": total_pixels,
            "valid_pixels": int(valid_count),
            "valid_ratio_%": round(valid_count / total_pixels * 100, 4),
            "min": float(np.nanmin(valid_values)),
            "max": float(np.nanmax(valid_values)),
            "mean": float(np.nanmean(valid_values)),
            "std": float(np.nanstd(valid_values)),
            "median": float(np.nanmedian(valid_values)),
            "p05": float(np.nanpercentile(valid_values, 5)),
            "p25": float(np.nanpercentile(valid_values, 25)),
            "p75": float(np.nanpercentile(valid_values, 75)),
            "p95": float(np.nanpercentile(valid_values, 95)),
        }

    ds.close()
    return stats

# =========================================================
# 4. 年度汇总统计
# =========================================================
def calc_year_stats(year, files):
    print("\n" + "=" * 90)
    print(f"📅 开始统计 {year} 年，共 {len(files)} 个文件")
    print("=" * 90)

    per_file_stats = []
    all_values = []

    total_pixels_sum = 0
    valid_pixels_sum = 0

    for i, f in enumerate(files, 1):
        try:
            s = calc_single_file_stats(f)
            per_file_stats.append(s)

            total_pixels_sum += s["total_pixels"]
            valid_pixels_sum += s["valid_pixels"]

            # 再读一次，仅提取有效值用于全年总体统计
            ds = xr.open_dataset(f)
            var_name = detect_var_name(ds)
            arr = np.squeeze(ds[var_name].values.astype(np.float64))
            valid_vals = arr[np.isfinite(arr)]
            if valid_vals.size > 0:
                all_values.append(valid_vals)
            ds.close()

            if i % 50 == 0 or i == len(files):
                print(f"  已完成: {i}/{len(files)}")

        except Exception as e:
            print(f"❌ 文件处理失败: {f}")
            print(f"   错误信息: {e}")

    # 年度整体统计
    if len(all_values) > 0:
        all_values = np.concatenate(all_values)

        year_stats = {
            "year": year,
            "file_count": len(per_file_stats),
            "total_pixels_sum": int(total_pixels_sum),
            "valid_pixels_sum": int(valid_pixels_sum),
            "valid_ratio_%": round(valid_pixels_sum / total_pixels_sum * 100, 4) if total_pixels_sum > 0 else np.nan,
            "min": float(np.nanmin(all_values)),
            "max": float(np.nanmax(all_values)),
            "mean": float(np.nanmean(all_values)),
            "std": float(np.nanstd(all_values)),
            "median": float(np.nanmedian(all_values)),
            "p05": float(np.nanpercentile(all_values, 5)),
            "p25": float(np.nanpercentile(all_values, 25)),
            "p75": float(np.nanpercentile(all_values, 75)),
            "p95": float(np.nanpercentile(all_values, 95)),
        }
    else:
        year_stats = {
            "year": year,
            "file_count": len(per_file_stats),
            "total_pixels_sum": int(total_pixels_sum),
            "valid_pixels_sum": int(valid_pixels_sum),
            "valid_ratio_%": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }

    return year_stats, per_file_stats

# =========================================================
# 5. 主程序
# =========================================================
def main():
    all_year_stats = []
    all_file_stats = []

    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        pattern = os.path.join(year_dir, "*.nc")
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            print(f"⚠️ {year} 年未找到 NC 文件: {year_dir}")
            continue

        year_stats, per_file_stats = calc_year_stats(year, files)

        all_year_stats.append(year_stats)

        for s in per_file_stats:
            s["year"] = year
            all_file_stats.append(s)

    # 转 DataFrame
    df_year = pd.DataFrame(all_year_stats)
    df_file = pd.DataFrame(all_file_stats)

    # 控制台输出年度统计
    print("\n" + "=" * 100)
    print("📊 2007-2011 年 ET-PET 年度整体统计")
    print("=" * 100)
    if not df_year.empty:
        print(df_year.to_string(index=False))
    else:
        print("没有可输出的年度统计结果。")

    # 每年每日有效像元数简单汇总
    if not df_file.empty:
        summary_daily = df_file.groupby("year").agg(
            文件数=("file", "count"),
            单日总像元_平均=("total_pixels", "mean"),
            单日有效像元_平均=("valid_pixels", "mean"),
            单日有效像元_最小=("valid_pixels", "min"),
            单日有效像元_最大=("valid_pixels", "max"),
            单日均值_平均=("mean", "mean"),
            单日均值_最小=("mean", "min"),
            单日均值_最大=("mean", "max"),
        ).reset_index()

        print("\n" + "=" * 100)
        print("📌 各年单日文件统计汇总")
        print("=" * 100)
        print(summary_daily.to_string(index=False))
    else:
        summary_daily = pd.DataFrame()

    # 保存 Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_year.to_excel(writer, sheet_name="年度整体统计", index=False)
        summary_daily.to_excel(writer, sheet_name="单日统计汇总", index=False)
        df_file.to_excel(writer, sheet_name="逐文件统计", index=False)

    print("\n" + "=" * 100)
    print(f"✅ 统计完成，结果已保存到：\n{output_excel}")
    print("=" * 100)


if __name__ == "__main__":
    main()