import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import regionmask
from scipy.stats import linregress
import rioxarray

# --- 1. 路径设置 ---
lst_dir = r'E:\data\meteorologicaldata\LST_8Day_MVC_2001_2022'
vci_path = r'E:\data\NDVI\VCI\VCI_8d_01deg_Wheat.nc'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
output_dir = r'E:\data\meteorologicaldata\IndexCalculationResults\TVDI'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_provincial_tvdi_wheat():
    print("🚀 开始小麦分省 TVDI 计算 (健壮性增强版)...")

    # 1. 加载并对齐 VCI 坐标名
    ds_vci = xr.open_dataset(vci_path)
    vci_da = ds_vci['VCI'].rename({'Lat': 'lat', 'Lon': 'lon'})

    # 2. 加载省界
    gdf = gpd.read_file(shp_path).reset_index(drop=True)
    print(f"检测到 {len(gdf)} 个行政区。")

    print("正在生成 2D 空间掩膜...")
    mask_2d = regionmask.mask_geopandas(gdf, vci_da.lon, vci_da.lat)

    lst_files = sorted(glob.glob(os.path.join(lst_dir, "*.tif")))
    final_tvdi = xr.full_like(vci_da, np.nan)

    # --- 3. 迭代时间轴 ---
    for i, t_val in enumerate(vci_da.time):
        date_dt = pd.to_datetime(t_val.values)
        date_str = date_dt.strftime('%Y_%m_%d')

        match = [f for f in lst_files if date_str in f]
        if not match:
            continue

        try:
            # 读取并重整 LST
            lst_raw = rioxarray.open_rasterio(match[0]).isel(band=0)
            lst_raw = lst_raw.rename({'x': 'lon', 'y': 'lat'})
            lst_step = lst_raw.reindex(lon=vci_da.lon, lat=vci_da.lat, method="nearest")

            vci_step = vci_da.isel(time=i)

            # 记录本期是否有任何省份拟合成功
            success_count = 0
            sample_r2 = 0.0
            last_prov = "None"

            # --- 4. 遍历省份拟合 ---
            present_regions = np.unique(mask_2d.values[~np.isnan(mask_2d.values)]).astype(int)

            for rid in present_regions:
                p_name = gdf.iloc[rid]['name']
                mask_sel = (mask_2d == rid)

                v_sub = vci_step.where(mask_sel)
                l_sub = lst_step.where(mask_sel)

                valid = (~np.isnan(v_sub)) & (~np.isnan(l_sub)) & (v_sub > 0)
                x_fit = v_sub.values[valid.values]
                y_fit = l_sub.values[valid.values]

                # 样本量阈值
                if len(x_fit) < 100:
                    continue

                # 提取干湿边界
                num_bins = 15
                bins = np.linspace(0.05, 0.95, num_bins + 1)
                dx, dy, wx, wy = [], [], [], []

                for b in range(num_bins):
                    b_idx = (x_fit >= bins[b]) & (x_fit < bins[b + 1])
                    if np.sum(b_idx) > 10:
                        dx.append(np.mean(x_fit[b_idx]))
                        dy.append(np.percentile(y_fit[b_idx], 98))
                        wx.append(np.mean(x_fit[b_idx]))
                        wy.append(np.percentile(y_fit[b_idx], 2))

                if len(dx) < 5:
                    continue

                # 回归计算
                s_d, i_d, r_d, _, _ = linregress(dx, dy)
                s_w, i_w, r_w, _, _ = linregress(wx, wy)

                # 计算 TVDI
                l_max = i_d + s_d * v_sub
                l_min = i_w + s_w * v_sub
                denom = (l_max - l_min).where(l_max - l_min > 0.1, 0.1)

                tvdi_prov = ((l_sub - l_min) / denom).clip(0, 1)

                # 更新结果
                final_tvdi.loc[dict(time=t_val)] = xr.where(mask_sel, tvdi_prov, final_tvdi.sel(time=t_val))

                success_count += 1
                sample_r2 = r_d ** 2
                last_prov = p_name

            # 每隔 20 期且有成功拟合时打印一次进度
            if i % 20 == 0:
                if success_count > 0:
                    print(f"✅ {date_str} 完成: {success_count} 省拟合成功 (示例: {last_prov} R2={sample_r2:.2f})")
                else:
                    print(f"ℹ️ {date_str}: 该期所有省份样本不足，跳过拟合。")

        except Exception as e:
            print(f"⚠️ 日期 {date_str} 发生非预期错误: {e}")
            continue

    # --- 5. 保存 ---
    final_tvdi = final_tvdi.rename({'lat': 'Lat', 'lon': 'Lon'})
    final_tvdi.name = "TVDI"
    output_path = os.path.join(output_dir, 'TVDI_8d_01deg_Wheat_Provincial.nc')
    final_tvdi.to_netcdf(output_path)
    print(f"\n🎉 运行结束！结果保存在: {output_path}")


if __name__ == "__main__":
    calculate_provincial_tvdi_wheat()