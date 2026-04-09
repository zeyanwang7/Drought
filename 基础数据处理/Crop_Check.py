import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
import os

# ===============================
# 1. 路径设置
# ===============================
spring_raw = r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Wheat_01deg.tif"
spring_new = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif"

winter_raw = r"E:\数据集\作物种植文件\0.1度重采样文件\Winter_Wheat_01deg.tif"
winter_new = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif"

shp_path = r"E:\数据集\中国_省\中国_省2.shp"

# ===============================
# 2. 读取省界
# ===============================
gdf = gpd.read_file(shp_path, encoding='UTF-8')

# 自动识别字段
name_field = None
for col in gdf.columns:
    if "name" in col.lower() or "省" in col:
        name_field = col
        break

if name_field is None:
    raise ValueError("❌ 未找到省名字段")

print(f"✅ 使用字段: {name_field}")


# ===============================
# 3. 核心函数：统计种植省份
# ===============================
def check_crop_provinces(raster_path):

    print("\n==============================")
    print(f"📂 检查文件: {os.path.basename(raster_path)}")

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform

    crop_mask = (data == 1)

    print(f"🌱 总种植像元数: {np.sum(crop_mask)}")

    province_stats = []

    for idx, row in gdf.iterrows():

        province_name = str(row[name_field])

        mask = rasterize(
            [(row.geometry, 1)],
            out_shape=data.shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        count = np.sum((mask == 1) & crop_mask)

        if count > 0:
            province_stats.append((province_name, int(count)))

    province_stats.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 有作物种植的省份：")
    for name, count in province_stats:
        print(f"{name:<10}  像元数: {count}")

    print(f"\n✅ 省份数量: {len(province_stats)}")

    return province_stats


# ===============================
# 4. 自动对比函数（核心新增）
# ===============================
def compare(before, after, label):

    before_set = set([x[0][:2] for x in before])
    after_set = set([x[0][:2] for x in after])

    removed = before_set - after_set
    remained = before_set & after_set

    print(f"\n================ {label} 删除效果检查 ================")
    print("✅ 成功删除的省份:", sorted(list(removed)))
    print("⚠️ 仍然存在的省份:", sorted(list(remained)))

    # 更严格检查（是否还有残留像元）
    print("\n🔍 进一步检查（是否有残留像元）:")
    for name, count in after:
        if name[:2] in removed:
            print(f"❌ {name} 仍有残留像元: {count}")


# ===============================
# 5. 执行检查
# ===============================
print("\n================= 春小麦（删除前） =================")
spring_before = check_crop_provinces(spring_raw)

print("\n================= 春小麦（删除后） =================")
spring_after = check_crop_provinces(spring_new)

compare(spring_before, spring_after, "春小麦")


print("\n================= 冬小麦（删除前） =================")
winter_before = check_crop_provinces(winter_raw)

print("\n================= 冬小麦（删除后） =================")
winter_after = check_crop_provinces(winter_new)

compare(winter_before, winter_after, "冬小麦")

print("\n🎉 检查完成！")