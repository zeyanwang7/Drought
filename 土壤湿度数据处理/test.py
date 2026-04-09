import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

fp = r"E:\data\Soil_Humidity_CropClip_ToCropGrid\Winter_Wheat\0_10cm\SMCI_9km_2000_0_10cm_Winter_Wheat.nc"

ds = xr.open_dataset(fp)
var_name = list(ds.data_vars)[0]
da = ds[var_name]

# 转成真实单位 m3/m3
da_real = da * 0.001
da_real.attrs["units"] = "m³/m³"

# 随机取一天
arr = da_real.isel(time=0)

# 若纬度降序则排序
if arr.lat.values[1] < arr.lat.values[0]:
    arr = arr.sortby("lat")

print("真实单位统计：")
print("min:", float(arr.min(skipna=True).values))
print("max:", float(arr.max(skipna=True).values))
print("mean:", float(arr.mean(skipna=True).values))

plt.figure(figsize=(8, 6), dpi=130)
plt.imshow(
    arr.values,
    origin="lower",
    extent=[float(arr.lon.min()), float(arr.lon.max()), float(arr.lat.min()), float(arr.lat.max())],
    cmap="viridis"
)
plt.colorbar(label="Soil Moisture (m³/m³)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Winter Wheat Soil Moisture (0-10 cm, real unit)")
plt.tight_layout()
plt.show()

ds.close()