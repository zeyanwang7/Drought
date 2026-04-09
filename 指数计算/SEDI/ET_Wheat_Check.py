import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ===============================
# 1. 路径
# ===============================
spring_mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif"
winter_mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif"

mask_paths = {
    "Spring_Wheat": spring_mask_path,
    "Winter_Wheat": winter_mask_path
}

# ===============================
# 2. 检查函数
# ===============================
def check_mask_info(mask_name, mask_path, do_plot=True):
    print("\n" + "=" * 90)
    print(f"📂 掩膜检查: {mask_name}")
    print("=" * 90)
    print(f"文件路径: {mask_path}")

    if not os.path.exists(mask_path):
        print("❌ 文件不存在")
        return

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        height, width = mask.shape

        # 根据transform构造像元中心点坐标
        lon = np.array([transform.c + (i + 0.5) * transform.a for i in range(width)], dtype=np.float64)
        lat = np.array([transform.f + (j + 0.5) * transform.e for j in range(height)], dtype=np.float64)

        print(f"shape: {mask.shape}")
        print(f"CRS: {crs}")
        print(f"bounds: {bounds}")
        print(f"transform: {transform}")
        print(f"分辨率: lon_res = {transform.a}, lat_res = {transform.e}")

        print(f"\n经度长度: {len(lon)}")
        print(f"纬度长度: {len(lat)}")
        print(f"经度范围: {lon.min()} ~ {lon.max()}")
        print(f"纬度范围: {lat.min()} ~ {lat.max()}")

        if len(lat) > 1:
            if lat[1] > lat[0]:
                print("✅ 纬度方向: 递增（南 -> 北）")
            else:
                print("⚠️ 纬度方向: 递减（北 -> 南）")
        else:
            print("⚠️ 纬度长度不足，无法判断方向")

        if len(lon) > 1:
            if lon[1] > lon[0]:
                print("✅ 经度方向: 递增（西 -> 东）")
            else:
                print("⚠️ 经度方向: 递减（东 -> 西）")

        # 统计有效像元
        valid_mask = np.isfinite(mask) & (mask != 0)
        valid_pixels = np.sum(valid_mask)
        total_pixels = mask.size

        print(f"\n总像元数: {total_pixels}")
        print(f"有效像元数(非0且非NaN): {valid_pixels}")
        print(f"无效/空白像元数: {total_pixels - valid_pixels}")

        # 如需统一为纬度递增，可提示后续是否flip
        if len(lat) > 1 and lat[1] < lat[0]:
            print("👉 后续匹配 xarray 的 lat 升序网格时，建议对 mask 执行 np.flipud(mask)，并令 lat = lat[::-1]")
        else:
            print("👉 当前 mask 已是纬度递增，通常无需翻转")

        # 绘图检查
        if do_plot:
            mask_plot = mask.copy()
            mask_plot[mask_plot == 0] = np.nan

            # 若纬度递减，为了 pycharm/imshow 正常显示，可转成递增
            plot_lat = lat.copy()
            plot_mask = mask_plot.copy()

            if len(plot_lat) > 1 and plot_lat[1] < plot_lat[0]:
                plot_lat = plot_lat[::-1]
                plot_mask = np.flipud(plot_mask)

            plt.figure(figsize=(8, 6))
            plt.imshow(
                plot_mask,
                origin="lower",
                extent=[lon.min(), lon.max(), plot_lat.min(), plot_lat.max()],
                cmap="YlGn"
            )
            plt.colorbar(label="Mask value")
            plt.title(f"{mask_name} 掩膜检查图")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.show()

            print("✅ 掩膜绘图完成")


# ===============================
# 3. 执行检查
# ===============================
for name, path in mask_paths.items():
    check_mask_info(name, path, do_plot=True)