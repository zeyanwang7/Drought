#该代码主要负责将20年的物候文件对每个像元取多年数据里的中位数
import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# =========================
# 输入路径
# =========================

input_folder = r"E:\数据集\作物物候\小麦"

# 输出路径
output_folder = r"E:\数据集\重采样后物候期\小麦"

median_file = os.path.join(output_folder, "Wheat_MA_median_2000_2019.tif")
class_file = os.path.join(output_folder, "Wheat_winter_spring_classification.tif")

files = sorted(glob.glob(os.path.join(input_folder, "CHN_Wheat_MA_*.tif")))

print("找到文件数量:", len(files))

stack = []

# =========================
# 读取参考栅格
# =========================

with rasterio.open(files[0]) as ref:

    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = ref.shape
    profile = ref.profile

# =========================
# 读取20年物候数据
# =========================

for f in files:

    print("处理:", os.path.basename(f))

    with rasterio.open(f) as src:

        data = src.read(1).astype(np.float32)

        # 处理 NoData
        data[data == src.nodata] = np.nan

        dst = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        stack.append(dst)

# =========================
# 转换为3D数组
# =========================

stack = np.stack(stack)

print("Stack shape:", stack.shape)

# =========================
# 计算逐像元 median
# =========================

median_ma = np.nanmedian(stack, axis=0)

# =========================
# 保存 median 栅格
# =========================

profile.update(dtype=rasterio.float32, nodata=np.nan)

with rasterio.open(median_file, "w", **profile) as dst:
    dst.write(median_ma.astype(np.float32), 1)

print("Median物候保存到:")
print(median_file)

# =========================
# 春 / 冬小麦分类
# =========================

threshold = 170

winter = np.where(median_ma <= threshold, 1, np.nan)
spring = np.where(median_ma > threshold, 2, np.nan)

classification = np.nan_to_num(winter) + np.nan_to_num(spring)

# =========================
# 保存分类栅格
# =========================

profile.update(dtype=rasterio.int16, nodata=0)

with rasterio.open(class_file, "w", **profile) as dst:
    dst.write(classification.astype(np.int16), 1)

print("分类栅格保存到:")
print(class_file)

# =========================
# 绘图
# =========================

cmap = ListedColormap([
    "white",
    "saddlebrown",
    "green"
])

plt.figure(figsize=(10,8))

plt.imshow(classification, cmap=cmap)

plt.title("Winter Maize (Brown) vs Spring Maize (Green)")
plt.xlabel("Column")
plt.ylabel("Row")

winter_patch = mpatches.Patch(color="saddlebrown", label="Summer Maize")
spring_patch = mpatches.Patch(color="green", label="Spring Maize")

plt.legend(handles=[winter_patch, spring_patch], loc="lower left")

plt.show()

# =========================
# 统计信息
# =========================

valid = median_ma[np.isfinite(median_ma)]

print("MA DOY statistics")
print("Min:", np.min(valid))
print("Max:", np.max(valid))
print("Mean:", np.mean(valid))