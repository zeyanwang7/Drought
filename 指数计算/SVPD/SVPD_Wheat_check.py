import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random

# ===============================
# 1. 文件路径（改这里）
# ===============================
file_path = r"E:\data\SVPD\Maize_test\SVPD_test.nc"

# ===============================
# 2. 读取数据
# ===============================
print("="*50)
print("📂 打开文件:")
print(file_path)

ds = xr.open_dataset(file_path)

var_name = list(ds.data_vars)[0]

data = ds[var_name].values
lat = ds["lat"].values
lon = ds["lon"].values
time = ds["time"].values

print("\n📌 数据形状:", data.shape)
print("📌 时间长度:", len(time))

# ===============================
# 3. 随机抽一天
# ===============================
idx = random.randint(0, len(time) - 1)

plot_data = data[idx]
plot_time = str(time[idx])[:10]

print("\n🎯 抽样日期:", plot_time)

# ===============================
# 4. 基本检查（强烈建议保留）
# ===============================
print("NaN比例:", np.isnan(plot_data).mean())
print("最小值:", np.nanmin(plot_data))
print("最大值:", np.nanmax(plot_data))

# ===============================
# 5. 绘图（不会翻转版本）
# ===============================
plt.figure(figsize=(10, 6))

img = plt.imshow(
    plot_data,
    origin="lower",   # 🔥必须（防翻转）
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]
)

plt.colorbar(img, label="SVPD")

plt.title(f"SVPD Random Check\n{plot_time}")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

# ===============================
# 6. 保存图片（可选）
# ===============================
save_path = file_path.replace(".nc", f"_check_{plot_time}.png")

plt.figure(figsize=(10, 6))
plt.imshow(
    plot_data,
    origin="lower",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]
)
plt.colorbar(label="SVPD")
plt.title(f"SVPD Check\n{plot_time}")
plt.savefig(save_path, dpi=150)
plt.close()

print("\n✅ 图片已保存:", save_path)