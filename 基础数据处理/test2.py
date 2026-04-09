#依据种植分布图将作物物候期分开，划分为春、夏玉米，冬、春小麦物候期
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 输入路径
# =========================

crop_folder = r"E:\数据集\作物种植文件\作物原始种植分布文件\小麦"

phen_folder = r"E:\数据集\重采样后物候期\小麦\1km"

output_folder = r"E:\数据集\重采样后物候期\小麦\1km"

os.makedirs(output_folder, exist_ok=True)


# =========================
# 种植分布文件
# =========================

winter_mask_file = os.path.join(
    crop_folder,
    "Winter_Wheat_Stable_2009_2015.tif"
)

spring_mask_file = os.path.join(
    crop_folder,
    "Spring_Wheat_Stable_2009_2015.tif"
)


# =========================
# 物候文件
# =========================

phen_files = {
    "GR_EM": os.path.join(phen_folder, "Wheat_GR&EM_median_2000_2019.tif"),
    "HE": os.path.join(phen_folder, "Wheat_HE_median_2000_2019.tif"),
    "MA": os.path.join(phen_folder, "Wheat_MA_median_2000_2019.tif")
}


# =========================
# 读取种植分布
# =========================

with rasterio.open(winter_mask_file) as src:

    winter_mask = src.read(1)

    transform = src.transform

    profile = src.profile


with rasterio.open(spring_mask_file) as src:

    spring_mask = src.read(1)


# =========================
# 处理物候
# =========================

for name, phen_file in phen_files.items():

    print("Processing:", name)

    with rasterio.open(phen_file) as src:

        phen = src.read(1).astype(float)

        phen[phen == src.nodata] = np.nan


    # 冬小麦物候
    winter_pheno = np.where(winter_mask > 0, phen, np.nan)

    # 春小麦物候
    spring_pheno = np.where(spring_mask > 0, phen, np.nan)


    # =========================
    # 保存栅格
    # =========================

    profile.update(dtype=rasterio.float32, nodata=-9999)


    winter_out = os.path.join(
        output_folder,
        f"Winter_Wheat_{name}.tif"
    )

    spring_out = os.path.join(
        output_folder,
        f"Spring_Wheat_{name}.tif"
    )


    w_data = winter_pheno.copy()
    w_data[np.isnan(w_data)] = -9999

    s_data = spring_pheno.copy()
    s_data[np.isnan(s_data)] = -9999


    with rasterio.open(winter_out, "w", **profile) as dst:

        dst.write(w_data.astype(np.float32), 1)


    with rasterio.open(spring_out, "w", **profile) as dst:

        dst.write(s_data.astype(np.float32), 1)


    print("Saved:", winter_out)
    print("Saved:", spring_out)


    # =========================
    # 绘图
    # =========================

    cmap = plt.cm.turbo.copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1,2, figsize=(14,6))

    axes[0].imshow(winter_pheno, cmap=cmap)
    axes[0].set_title(f"Winter Wheat {name}")

    axes[1].imshow(spring_pheno, cmap=cmap)
    axes[1].set_title(f"Spring Wheat {name}")

    for ax in axes:
        ax.axis("off")

    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=axes,
        fraction=0.03,
        pad=0.04,
        label="DOY"
    )

    plt.show()


print("\nAll wheat phenology processed")