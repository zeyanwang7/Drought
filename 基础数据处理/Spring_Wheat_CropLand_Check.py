import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

# ===============================
# 1. 路径设置
# ===============================
spring_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Wheat_01deg.tif"
winter_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Winter_Wheat_01deg.tif"
shp_path = r"E:\数据集\中国_省\中国_省2.shp"

output_dir = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版"
os.makedirs(output_dir, exist_ok=True)

out_spring = os.path.join(output_dir, "Spring_Wheat_01deg_filtered.tif")
out_winter = os.path.join(output_dir, "Winter_Wheat_01deg_filtered.tif")

# ===============================
# 2. 删除省份配置
# ===============================
spring_remove = ["河南", "陕西", "四川", "湖北", "山西", "天津", "北京","云南"]
winter_remove = ["广西", "青海","内蒙","黑龙江","吉林"]

# ===============================
# 3. 读取省界
# ===============================
print("📌 读取省界数据...")
gdf = gpd.read_file(shp_path, encoding='UTF-8')

print("字段列表:", gdf.columns)

# 自动识别省名字段
name_field = None
for col in gdf.columns:
    if "name" in col.lower() or "省" in col:
        name_field = col
        break

if name_field is None:
    raise ValueError("❌ 未找到省名字段，请手动指定")

print(f"✅ 使用省名字段: {name_field}")

# ===============================
# 4. 前两个字匹配函数
# ===============================
def get_target_gdf(gdf, provinces):
    def match(name):
        if name is None:
            return False
        name_str = str(name)
        return any(name_str[:2] == p[:2] for p in provinces)

    selected = gdf[gdf[name_field].apply(match)]
    print("🎯 匹配到省份:", selected[name_field].values)
    return selected


# ===============================
# 5. 核心处理函数
# ===============================
def process_crop(raster_path, output_path, provinces):

    print(f"\n🚀 开始处理: {os.path.basename(raster_path)}")

    gdf_remove = get_target_gdf(gdf, provinces)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        meta = src.meta.copy()

        print(f"📐 栅格大小: {data.shape}")
        print(f"📊 原始种植像元数: {np.sum(data == 1)}")

        # 栅格化省界
        print("🧩 栅格化目标省份...")
        mask = rasterize(
            [(geom, 1) for geom in gdf_remove.geometry],
            out_shape=data.shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        # 删除像元
        result = data.copy()
        remove_area = (mask == 1) & (data == 1)

        print(f"🧹 待删除像元数量: {np.sum(remove_area)}")

        nodata_value = 0
        result[remove_area] = nodata_value

        print(f"📊 删除后种植像元数: {np.sum(result == 1)}")

        # 保存
        print("💾 保存结果...")
        meta.update({
            "dtype": "uint8",
            "nodata": nodata_value,
            "compress": "lzw"
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result, 1)

        print(f"✅ 输出完成: {output_path}")


# ===============================
# 6. 执行
# ===============================
print("\n================= 开始处理 =================")

# 春小麦
process_crop(spring_path, out_spring, spring_remove)

# 冬小麦
process_crop(winter_path, out_winter, winter_remove)

print("\n🎉 全部处理完成！")