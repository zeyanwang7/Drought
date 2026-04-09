import os
import re
import xarray as xr
import numpy as np

# ===============================
# 1. 路径设置
# ===============================
tmp_root = r"E:\data\meteorologicaldata\Tmpmax"
rh_root = r"E:\data\meteorologicaldata\RelativeHumidity"
output_root = r"E:\data\meteorologicaldata\VPD"

start_year = 2000
end_year = 2022


# ===============================
# 2. 自动识别变量名
# ===============================
def guess_var_name(ds, keywords=None):
    """
    自动识别数据变量名
    """
    data_vars = list(ds.data_vars)
    if keywords is None:
        keywords = []

    for kw in keywords:
        for v in data_vars:
            if kw.lower() in v.lower():
                return v

    # 如果没找到，就返回第一个变量
    if len(data_vars) > 0:
        return data_vars[0]

    raise ValueError("❌ 未找到数据变量！")


# ===============================
# 3. 饱和水汽压计算函数
# ===============================
def calc_es_from_temp(temp_c):
    """
    根据气温(℃)计算饱和水汽压 es(kPa)
    FAO-56 公式
    """
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


# ===============================
# 4. 计算单日VPDmax
# ===============================
def calc_vpdmax(tmax, rh):
    """
    tmax: 最高气温 (℃)
    rh  : 相对湿度 (%)

    返回:
    vpdmax (kPa)
    """
    es = calc_es_from_temp(tmax)

    # 避免异常RH
    rh = np.clip(rh, 0, 100)

    vpdmax = es * (1.0 - rh / 100.0)

    # 避免负值
    vpdmax = xr.where(vpdmax < 0, 0, vpdmax)

    return vpdmax


# ===============================
# 5. 从文件名提取日期
# ===============================
def extract_date_from_filename(filename):
    """
    从文件名中提取 YYYY_MM_DD
    例如:
    ChinaMet_010deg_tmpmax_2000_01_01.nc
    """
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    if m:
        y, mm, dd = m.groups()
        return f"{y}-{mm}-{dd}"
    else:
        raise ValueError(f"❌ 无法从文件名提取日期: {filename}")


# ===============================
# 6. 主计算程序
# ===============================
def calculate_daily_vpdmax():
    print("=" * 60)
    print("🚀 开始计算每日最大VPD (VPDmax)")
    print("=" * 60)

    for year in range(start_year, end_year + 1):
        print(f"\n📅 正在处理年份: {year}")

        tmp_dir = os.path.join(tmp_root, str(year))
        rh_dir = os.path.join(rh_root, str(year))
        out_dir = os.path.join(output_root, str(year))

        if not os.path.exists(tmp_dir):
            print(f"⚠️ 温度文件夹不存在，跳过: {tmp_dir}")
            continue

        if not os.path.exists(rh_dir):
            print(f"⚠️ 相对湿度文件夹不存在，跳过: {rh_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        tmp_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith(".nc")])
        rh_files = sorted([f for f in os.listdir(rh_dir) if f.endswith(".nc")])

        print(f"  最高温文件数: {len(tmp_files)}")
        print(f"  相对湿度文件数: {len(rh_files)}")

        # 构建湿度文件字典：日期 -> 文件名
        rh_dict = {}
        for rf in rh_files:
            try:
                date_str = extract_date_from_filename(rf)
                rh_dict[date_str] = rf
            except Exception as e:
                print(f"  ⚠️ 湿度文件跳过: {rf} | 原因: {e}")

        processed = 0
        skipped = 0

        for i, tf in enumerate(tmp_files, start=1):
            try:
                date_str = extract_date_from_filename(tf)
            except Exception as e:
                print(f"  ⚠️ 温度文件跳过: {tf} | 原因: {e}")
                skipped += 1
                continue

            if date_str not in rh_dict:
                print(f"  ⚠️ 缺少对应湿度文件: {date_str}")
                skipped += 1
                continue

            tmp_path = os.path.join(tmp_dir, tf)
            rh_path = os.path.join(rh_dir, rh_dict[date_str])

            out_name = f"ChinaMet_010deg_VPDmax_{date_str.replace('-', '_')}.nc"
            out_path = os.path.join(out_dir, out_name)

            # 若已存在可跳过
            if os.path.exists(out_path):
                if i % 50 == 0 or i == len(tmp_files):
                    print(f"  ... 进度 {i}/{len(tmp_files)} (已存在跳过)")
                continue

            try:
                ds_t = xr.open_dataset(tmp_path)
                ds_rh = xr.open_dataset(rh_path)

                # 自动识别变量
                t_var = guess_var_name(ds_t, keywords=["tmpmax", "tmax", "temp", "tmp"])
                rh_var = guess_var_name(ds_rh, keywords=["rhu", "rh", "humidity"])

                tmax = ds_t[t_var]
                rh = ds_rh[rh_var]

                # 如果维度名字不统一，可在这里做重命名
                rename_dict_t = {}
                rename_dict_rh = {}

                if "longitude" in tmax.dims:
                    rename_dict_t["longitude"] = "lon"
                if "latitude" in tmax.dims:
                    rename_dict_t["latitude"] = "lat"

                if "longitude" in rh.dims:
                    rename_dict_rh["longitude"] = "lon"
                if "latitude" in rh.dims:
                    rename_dict_rh["latitude"] = "lat"

                if rename_dict_t:
                    tmax = tmax.rename(rename_dict_t)
                if rename_dict_rh:
                    rh = rh.rename(rename_dict_rh)

                # 对齐网格
                tmax, rh = xr.align(tmax, rh, join="exact")

                # 计算VPDmax
                vpdmax = calc_vpdmax(tmax, rh).astype(np.float32)

                # 设置变量名与属性
                vpdmax.name = "VPDmax"
                vpdmax.attrs = {
                    "long_name": "Daily Maximum Vapor Pressure Deficit",
                    "units": "kPa",
                    "description": "Calculated from daily maximum temperature and relative humidity using FAO-56 saturation vapor pressure equation"
                }

                # 转为Dataset
                ds_out = vpdmax.to_dataset()

                ds_out.attrs = {
                    "title": "Daily Maximum Vapor Pressure Deficit",
                    "source_temperature": tmp_path,
                    "source_relative_humidity": rh_path,
                    "date": date_str,
                    "formula": "VPDmax = es(Tmax) * (1 - RH/100)",
                    "es_formula": "es = 0.6108 * exp(17.27*T / (T + 237.3))"
                }

                # 压缩保存
                encoding = {
                    "VPDmax": {
                        "dtype": "float32",
                        "zlib": True,
                        "complevel": 4
                    }
                }

                ds_out.to_netcdf(out_path, encoding=encoding)

                ds_t.close()
                ds_rh.close()
                ds_out.close()

                processed += 1

                if i % 30 == 0 or i == len(tmp_files):
                    print(f"  ... 进度 {i}/{len(tmp_files)} | 已完成 {processed} | 跳过 {skipped}")

            except Exception as e:
                print(f"  ❌ 处理失败: {tf}")
                print(f"     原因: {e}")
                skipped += 1

        print(f"✅ {year} 年处理完成：成功 {processed} 个，跳过 {skipped} 个")

    print("\n" + "=" * 60)
    print("🎉 全部年份 VPDmax 计算完成！")
    print("=" * 60)


# ===============================
# 7. 运行
# ===============================
if __name__ == "__main__":
    calculate_daily_vpdmax()