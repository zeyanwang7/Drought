import os
import numpy as np
import rasterio

# ===============================
# 1. 路径设置
# ===============================
base_dir = r"E:\数据集\重采样后物候期\小麦\0.1度_修复版"

files = {

    "Winter_Wheat_GR_EM": os.path.join(base_dir, "Winter_Wheat_GR_EM_0.1deg_fixed.tif"),
    "Winter_Wheat_HE":    os.path.join(base_dir, "Winter_Wheat_HE_0.1deg_fixed.tif"),
    "Winter_Wheat_MA":    os.path.join(base_dir, "Winter_Wheat_MA_0.1deg_fixed.tif"),
}

# ===============================
# 2. 读取并检查单个 tif
# ===============================
def check_single_tif(name, tif_path):
    print("\n" + "=" * 100)
    print(f"📂 文件: {name}")
    print(tif_path)

    if not os.path.exists(tif_path):
        print("❌ 文件不存在！")
        return None, None

    with rasterio.open(tif_path) as src:
        data = src.read(1)

        print("\n📌 基本信息")
        print(f"shape      : {data.shape}")
        print(f"dtype      : {data.dtype}")
        print(f"crs        : {src.crs}")
        print(f"bounds     : {src.bounds}")
        print(f"transform  : {src.transform}")
        print(f"resolution : {src.res}")
        print(f"nodata     : {src.nodata}")

        # 转 float，便于赋 NaN
        arr = data.astype(np.float32)

        # nodata 处理
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        # 常见无效值再额外处理一下
        invalid_values = [-9999, -999, 0]
        for v in invalid_values:
            arr[arr == v] = np.nan

        valid = arr[np.isfinite(arr)]

        print("\n📌 DOY统计")
        total_pixels = arr.size
        valid_pixels = valid.size
        nan_pixels = total_pixels - valid_pixels

        print(f"总像元数     : {total_pixels}")
        print(f"有效像元数   : {valid_pixels}")
        print(f"无效像元数   : {nan_pixels}")

        if valid_pixels > 0:
            print(f"最小值       : {np.min(valid):.2f}")
            print(f"最大值       : {np.max(valid):.2f}")
            print(f"均值         : {np.mean(valid):.2f}")
            print(f"中位数       : {np.median(valid):.2f}")
            print(f"标准差       : {np.std(valid):.2f}")

            # 唯一值个数（若是整数DOY通常不算特别多）
            unique_vals = np.unique(valid.astype(np.int32))
            print(f"唯一DOY个数  : {len(unique_vals)}")
            print(f"前20个唯一值 : {unique_vals[:20]}")
            print(f"后20个唯一值 : {unique_vals[-20:]}")
        else:
            print("⚠️ 没有有效像元！")

        # 检查异常 DOY
        if valid_pixels > 0:
            bad_mask = (valid < 1) | (valid > 366)
            bad_count = np.sum(bad_mask)
            print(f"\n📌 异常DOY像元数(<1 或 >366): {bad_count}")

        return arr, src.profile


# ===============================
# 3. 检查三期顺序关系
# ===============================
def check_stage_order(crop_name, gr_em, he, ma):
    print("\n" + "#" * 100)
    print(f"🔍 {crop_name} 三个物候期顺序检查")
    print("#" * 100)

    # 三者都有效的像元
    valid_mask = np.isfinite(gr_em) & np.isfinite(he) & np.isfinite(ma)
    valid_count = np.sum(valid_mask)

    print(f"三期同时有效像元数: {valid_count}")

    if valid_count == 0:
        print("⚠️ 没有三期同时有效的像元，无法检查顺序。")
        return

    gr_em_v = gr_em[valid_mask]
    he_v = he[valid_mask]
    ma_v = ma[valid_mask]

    # 春小麦/一般情况：GR&EM < HE < MA
    cond_all_right = (gr_em_v < he_v) & (he_v < ma_v)
    cond_gr_he_bad = ~(gr_em_v < he_v)
    cond_he_ma_bad = ~(he_v < ma_v)
    cond_gr_ma_bad = ~(gr_em_v < ma_v)

    print(f"满足 GR&EM < HE < MA 的像元数: {np.sum(cond_all_right)}")
    print(f"不满足 GR&EM < HE 的像元数   : {np.sum(cond_gr_he_bad)}")
    print(f"不满足 HE < MA 的像元数      : {np.sum(cond_he_ma_bad)}")
    print(f"不满足 GR&EM < MA 的像元数   : {np.sum(cond_gr_ma_bad)}")

    # 阶段长度统计
    stage1 = he_v - gr_em_v
    stage2 = ma_v - he_v
    total_stage = ma_v - gr_em_v

    print("\n📌 阶段长度统计（单位：天）")
    print(f"阶段1(HE-GR&EM) 平均: {np.nanmean(stage1):.2f}, 最小: {np.nanmin(stage1):.2f}, 最大: {np.nanmax(stage1):.2f}")
    print(f"阶段2(MA-HE)    平均: {np.nanmean(stage2):.2f}, 最小: {np.nanmin(stage2):.2f}, 最大: {np.nanmax(stage2):.2f}")
    print(f"总生育期(MA-GR&EM)平均: {np.nanmean(total_stage):.2f}, 最小: {np.nanmin(total_stage):.2f}, 最大: {np.nanmax(total_stage):.2f}")

    # 随机抽样几个像元
    idx = np.argwhere(valid_mask)
    sample_n = min(10, len(idx))
    if sample_n > 0:
        np.random.seed(42)
        sample_idx = idx[np.random.choice(len(idx), sample_n, replace=False)]

        print("\n📌 随机抽样像元的三期DOY值")
        for k, (r, c) in enumerate(sample_idx, 1):
            print(f"样本{k:02d} | row={r}, col={c} | GR&EM={gr_em[r, c]:.1f}, HE={he[r, c]:.1f}, MA={ma[r, c]:.1f}")


# ===============================
# 4. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 100)
    print("开始检查小麦物候期 DOY 栅格文件")
    print("=" * 100)

    results = {}

    # 先逐个检查
    for name, path in files.items():
        arr, profile = check_single_tif(name, path)
        results[name] = arr

   

    # 再检查冬小麦顺序
    if (results["Winter_Wheat_GR_EM"] is not None and
        results["Winter_Wheat_HE"] is not None and
        results["Winter_Wheat_MA"] is not None):
        check_stage_order(
            "Winter_Wheat",
            results["Winter_Wheat_GR_EM"],
            results["Winter_Wheat_HE"],
            results["Winter_Wheat_MA"]
        )

    print("\n" + "=" * 100)
    print("✅ 检查完成")
    print("=" * 100)