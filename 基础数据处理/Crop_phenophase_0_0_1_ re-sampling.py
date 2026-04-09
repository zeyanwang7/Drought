import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling

# =============================
# 输入输出路径
# =============================

input_folder = r"E:\数据集\重采样后物候期\小麦\1km"
output_folder = r"E:\数据集\重采样后物候期\小麦\0.01度"

os.makedirs(output_folder, exist_ok=True)

files = glob.glob(os.path.join(input_folder, "*.tif"))

print("发现物候文件数量:", len(files))

# =============================
# 批量处理
# =============================

for src_file in files:

    name = os.path.basename(src_file)

    dst_file = os.path.join(
        output_folder,
        name.replace(".tif", "_0.01deg.tif")
    )

    print("\nProcessing:", name)

    with rasterio.open(src_file) as src:

        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
            resolution=0.01
        )

        profile = src.profile.copy()

        profile.update({
            "crs": "EPSG:4326",
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_file, "w", **profile) as dst:

            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )

    print("Saved:", dst_file)

    # =============================
    # 绘图检查
    # =============================

    with rasterio.open(dst_file) as src:

        data = src.read(1).astype(float)

        nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan


    cmap = plt.cm.turbo.copy()
    cmap.set_bad("white")

    plt.figure(figsize=(8,5))

    plt.imshow(data, cmap=cmap)

    plt.title(name.replace(".tif",""))

    plt.colorbar(label="DOY")

    plt.axis("off")

    plt.show()


print("\n所有物候文件处理完成")