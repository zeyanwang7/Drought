import os
import numpy as np
import rasterio

# ===============================
# 1. 路径设置
# ===============================
base_dir = r"E:\数据集\重采样后物候期\玉米\0.1度"
out_dir = r"E:\数据集\重采样后物候期\玉米\0.1度_修复版"

os.makedirs(out_dir, exist_ok=True)


# ===============================
# 2. 读取栅格
# ===============================
def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, profile


# ===============================
# 3. 输出顺序统计
# ===============================
def print_order_stats(title, v3, he, ma):
    """
    玉米三期顺序关系检查：
    V3 < HE < MA
    """
    valid_all = np.isfinite(v3) & np.isfinite(he) & np.isfinite(ma)
    valid_triplet = valid_all & (v3 < he) & (he < ma)

    bad_v3_he = valid_all & ~(v3 < he)
    bad_he_ma = valid_all & ~(he < ma)
    bad_any = valid_all & ~((v3 < he) & (he < ma))

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"三期同时有效像元数: {np.sum(valid_all)}")
    print(f"满足 V3 < HE < MA 的像元数: {np.sum(valid_triplet)}")
    print(f"不满足 V3 < HE 的像元数  : {np.sum(bad_v3_he)}")
    print(f"不满足 HE < MA 的像元数  : {np.sum(bad_he_ma)}")
    print(f"总异常像元数             : {np.sum(bad_any)}")

    if np.sum(valid_triplet) > 0:
        stage1 = he[valid_triplet] - v3[valid_triplet]
        stage2 = ma[valid_triplet] - he[valid_triplet]
        total = ma[valid_triplet] - v3[valid_triplet]

        print("\n阶段长度统计（仅正常像元）")
        print(f"阶段1(HE-V3) 平均: {np.mean(stage1):.2f}, 最小: {np.min(stage1):.2f}, 最大: {np.max(stage1):.2f}")
        print(f"阶段2(MA-HE) 平均: {np.mean(stage2):.2f}, 最小: {np.min(stage2):.2f}, 最大: {np.max(stage2):.2f}")
        print(f"总生育期(MA-V3)平均: {np.mean(total):.2f}, 最小: {np.min(total):.2f}, 最大: {np.max(total):.2f}")


# ===============================
# 4. 邻域联合修复
# ===============================
def repair_bad_pixels(v3, he, ma, max_radius=4, min_neighbors=5):
    """
    对异常像元做联合修复：
    - 只修复三期都有效但不满足 V3 < HE < MA 的像元
    - 使用邻域内正常像元的三期中位数联合赋值
    - radius=1 表示 3x3, radius=2 表示 5x5, ...
    """
    v3_new = v3.copy()
    he_new = he.copy()
    ma_new = ma.copy()

    nrows, ncols = v3.shape

    valid_all = np.isfinite(v3) & np.isfinite(he) & np.isfinite(ma)
    good = valid_all & (v3 < he) & (he < ma)
    bad = valid_all & ~((v3 < he) & (he < ma))

    bad_idx = np.argwhere(bad)

    print(f"\n待修复异常像元数: {len(bad_idx)}")

    repaired_count = 0
    failed_count = 0

    for k, (r, c) in enumerate(bad_idx, 1):
        repaired = False

        for radius in range(1, max_radius + 1):
            r0 = max(0, r - radius)
            r1 = min(nrows, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(ncols, c + radius + 1)

            good_win = good[r0:r1, c0:c1]
            if np.sum(good_win) < min_neighbors:
                continue

            v3_vals = v3[r0:r1, c0:c1][good_win]
            he_vals = he[r0:r1, c0:c1][good_win]
            ma_vals = ma[r0:r1, c0:c1][good_win]

            # 中位数联合赋值
            v3_fill = np.rint(np.nanmedian(v3_vals))
            he_fill = np.rint(np.nanmedian(he_vals))
            ma_fill = np.rint(np.nanmedian(ma_vals))

            # 检查顺序
            if np.isfinite(v3_fill) and np.isfinite(he_fill) and np.isfinite(ma_fill):
                if (v3_fill < he_fill) and (he_fill < ma_fill):
                    v3_new[r, c] = v3_fill
                    he_new[r, c] = he_fill
                    ma_new[r, c] = ma_fill
                    repaired = True
                    repaired_count += 1
                    break

        if not repaired:
            v3_new[r, c] = np.nan
            he_new[r, c] = np.nan
            ma_new[r, c] = np.nan
            failed_count += 1

        if (k % 50 == 0) or (k == len(bad_idx)):
            print(f"修复进度: {k}/{len(bad_idx)} | 已修复: {repaired_count} | 失败置NaN: {failed_count}")

    return v3_new, he_new, ma_new


# ===============================
# 5. 保存栅格
# ===============================
def save_tif(path, arr, profile):
    profile_out = profile.copy()
    profile_out.update(
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32), 1)


# ===============================
# 6. 单个作物处理函数
# ===============================
def process_crop(crop_name):
    """
    crop_name:
        Spring_Maize
        Summer_Maize
    """
    print("\n" + "#" * 100)
    print(f"开始处理作物: {crop_name}")
    print("#" * 100)

    v3_path = os.path.join(base_dir, f"{crop_name}_V3_0.1deg.tif")
    he_path = os.path.join(base_dir, f"{crop_name}_HE_0.1deg.tif")
    ma_path = os.path.join(base_dir, f"{crop_name}_MA_0.1deg.tif")

    out_v3 = os.path.join(out_dir, f"{crop_name}_V3_0.1deg_fixed.tif")
    out_he = os.path.join(out_dir, f"{crop_name}_HE_0.1deg_fixed.tif")
    out_ma = os.path.join(out_dir, f"{crop_name}_MA_0.1deg_fixed.tif")

    # 检查文件是否存在
    for p in [v3_path, he_path, ma_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    # 读取
    v3, profile = read_tif(v3_path)
    he, _ = read_tif(he_path)
    ma, _ = read_tif(ma_path)

    # 若有小数，先四舍五入为整数DOY
    v3 = np.where(np.isfinite(v3), np.rint(v3), np.nan)
    he = np.where(np.isfinite(he), np.rint(he), np.nan)
    ma = np.where(np.isfinite(ma), np.rint(ma), np.nan)

    # 修复前统计
    print_order_stats(f"{crop_name} 修复前顺序检查", v3, he, ma)

    # 修复
    v3_fixed, he_fixed, ma_fixed = repair_bad_pixels(
        v3, he, ma,
        max_radius=4,      # 最大扩展到9x9邻域
        min_neighbors=5    # 至少5个正常邻居
    )

    # 修复后统计
    print_order_stats(f"{crop_name} 修复后顺序检查", v3_fixed, he_fixed, ma_fixed)

    # 保存
    save_tif(out_v3, v3_fixed, profile)
    save_tif(out_he, he_fixed, profile)
    save_tif(out_ma, ma_fixed, profile)

    print("\n✅ 处理完成，结果已保存：")
    print(out_v3)
    print(out_he)
    print(out_ma)


# ===============================
# 7. 主程序
# ===============================
if __name__ == "__main__":
    print("=" * 100)
    print("开始修复春玉米 / 夏玉米物候异常像元（邻域联合中位数法）")
    print("=" * 100)

    crop_list = [
        "Spring_Maize",
        "Summer_Maize"
    ]

    for crop in crop_list:
        process_crop(crop)

    print("\n" + "=" * 100)
    print("✅ 全部处理完成")
    print("=" * 100)