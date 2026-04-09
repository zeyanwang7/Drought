import rioxarray
import os
import glob

# 修改为你的 LST TIF 文件夹路径
lst_dir = r'E:\data\meteorologicaldata\LST_8Day_MVC_2001_2022'


def check_tif_dimensions(directory):
    # 获取第一个 TIF 文件
    tif_files = glob.glob(os.path.join(directory, "*.tif"))
    if not tif_files:
        print("❌ 未在指定目录找到 TIF 文件，请检查路径。")
        return

    test_file = tif_files[0]
    print(f"🔍 正在读取测试文件: {os.path.basename(test_file)}")

    # 使用 rioxarray 打开
    rds = rioxarray.open_rasterio(test_file)

    print("\n--- 1. 维度名称 (Dimensions) ---")
    print(rds.dims)

    print("\n--- 2. 坐标名称 (Coordinates) ---")
    print(rds.coords)

    print("\n--- 3. 空间投影信息 (CRS) ---")
    print(rds.rio.crs)

    print("\n--- 4. 分辨率 (Resolution) ---")
    print(f"x方向分辨率: {rds.rio.transform()[0]}")
    print(f"y方向分辨率: {rds.rio.transform()[4]}")


if __name__ == "__main__":
    check_tif_dimensions(lst_dir)