import os
import random
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 路径设置
# ===============================
svpd_root = r"E:\data\SVPD\Wheat"

# 如果你想叠加中国省界/国界，可以填写shp路径；否则设为None
shp_path = r"E:\数据集\中国_省\中国_省2.shp"
# shp_path = None

# ===============================
# 2. 文件配置
# ===============================
crop_list = ["Winter_Wheat", "Spring_Wheat"]
scale_list = ["1month", "3month", "6month", "12month"]

# ===============================
# 3. 绘图参数
# ===============================
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文显示
plt.rcParams["axes.unicode_minus"] = False     # 负号显示

# SVPD一般可先用这个范围看图，后续可按实际再改
VMIN = -3
VMAX = 3

# ===============================
# 4. 可选加载边界
# ===============================
gdf = None
if shp_path is not None and os.path.exists(shp_path):
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        print(f"✅ 成功读取边界文件: {shp_path}")
    except Exception as e:
        print(f"⚠️ 边界文件读取失败，将不叠加边界: {e}")
        gdf = None
else:
    print("⚠️ 未使用shp边界叠加")

# ===============================
# 5. 循环绘图
# ===============================
for crop in crop_list:
    for scale in scale_list:
        nc_path = os.path.join(svpd_root, f"SVPD_{scale}_{crop}.nc")

        print("\n" + "=" * 70)
        print(f"📂 正在检查文件: {nc_path}")

        if not os.path.exists(nc_path):
            print("❌ 文件不存在，跳过")
            continue

        try:
            ds = xr.open_dataset(nc_path)

            if "SVPD" not in ds.data_vars:
                print("❌ 未找到变量 SVPD")
                ds.close()
                continue

            da = ds["SVPD"]

            print(ds)
            print(f"变量维度: {da.dims}")
            print(f"变量形状: {da.shape}")

            # -------------------------------
            # 自动统一维度顺序为 time, lat, lon
            # -------------------------------
            target_dims = [d for d in ["time", "lat", "lon"] if d in da.dims]
            da = da.transpose(*target_dims)

            if not all(d in da.dims for d in ["time", "lat", "lon"]):
                print("❌ 变量维度不包含 time/lat/lon，跳过")
                ds.close()
                continue

            time_vals = ds["time"].values
            lat_vals = ds["lat"].values
            lon_vals = ds["lon"].values

            # -------------------------------
            # 随机抽取一天
            # -------------------------------
            ntime = da.sizes["time"]
            rand_idx = random.randint(0, ntime - 1)

            da_day = da.isel(time=rand_idx)
            plot_data = da_day.values.astype(np.float32)

            date_str = str(np.datetime_as_string(time_vals[rand_idx], unit="D"))

            print(f"🎯 随机抽取日期: {date_str}")
            print(f"数值范围: min={np.nanmin(plot_data):.3f}, max={np.nanmax(plot_data):.3f}, mean={np.nanmean(plot_data):.3f}")

            # -------------------------------
            # 处理坐标方向
            # imshow中，若lat递增，则用origin='lower'
            # 若lat递减，则用origin='upper'
            # -------------------------------
            if lat_vals[0] < lat_vals[-1]:
                origin = "lower"
                extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]
                print("✅ 纬度方向: 递增，使用 origin='lower'")
            else:
                origin = "upper"
                extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]
                print("✅ 纬度方向: 递减，使用 origin='upper'")

            # -------------------------------
            # 绘图
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 7))

            im = ax.imshow(
                plot_data,
                extent=extent,
                origin=origin,
                cmap="RdBu_r",
                vmin=VMIN,
                vmax=VMAX,
                interpolation="nearest"
            )

            # 叠加边界
            if gdf is not None:
                try:
                    gdf.boundary.plot(ax=ax, linewidth=0.5, color="black")
                except Exception as e:
                    print(f"⚠️ 边界叠加失败: {e}")

            ax.set_title(f"{crop} | {scale} | {date_str}", fontsize=14)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            cbar = plt.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label("SVPD")

            plt.tight_layout()
            plt.show()

            ds.close()

        except Exception as e:
            print(f"❌ 绘图失败: {e}")