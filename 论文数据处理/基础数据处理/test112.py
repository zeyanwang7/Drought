import geopandas as gpd
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
china_map = gpd.read_file(shp_path, encoding='utf-8')
# 看看前10个地图里的原始名字
print("SHP原始名称样例：")
print(china_map['name'].head(10).tolist())