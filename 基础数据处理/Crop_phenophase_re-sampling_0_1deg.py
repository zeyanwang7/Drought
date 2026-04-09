import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt

# 路径设置
mask_path = r'E:\数据集\作物种植文件\0.1度重采样文件\Winter_Wheat_0p1deg_mask.tif'
pheno_dir = r'E:\数据集\重采样后物候期\小麦\0.01度'
output_dir = r'E:\数据集\重采样后物候期\小麦\0.1度'

pheno_files = ['Winter_Wheat_GR_EM_0.01deg.tif', 'Winter_Wheat_HE_0.01deg.tif', 'Winter_Wheat_MA_0.01deg.tif']


def smart_resample():
    # 1. 打开 0.1度 MASK 获取标准网格
    with rasterio.open(mask_path) as m_src:
        mask_data = m_src.read(1)
        m_meta = m_src.meta.copy()
        m_transform = m_src.transform  # 0.1度转换矩阵
        rows_01, cols_01 = m_src.shape

    results = {}

    for f_name in pheno_files:
        print(f"正在处理: {f_name}")
        p_path = os.path.join(pheno_dir, f_name)

        with rasterio.open(p_path) as p_src:
            p_data = p_src.read(1).astype(np.float32)
            p_nodata = p_src.nodata
            p_transform = p_src.transform  # 0.01度转换矩阵

            # 处理无效值
            if p_nodata is not None:
                p_data[p_data == p_nodata] = np.nan
            p_data[p_data <= 0] = np.nan

            # 初始化输出
            out_array = np.full((rows_01, cols_01), np.nan, dtype=np.float32)

            # 2. 遍历 0.1度的每一个像素点
            for r in range(rows_01):
                for c in range(cols_01):
                    if mask_data[r, c] > 0:
                        # 获取 0.1度像素中心点的经纬度
                        lon, lat = m_src.xy(r, c)

                        # 找到该经纬度在 0.01度图中对应的行列索引
                        # 0.1度像素覆盖的范围大约是中心点上下左右各 0.05度
                        # 我们取该范围内的 10x10 块
                        py, px = p_src.index(lon - 0.045, lat + 0.045)  # 左上角索引

                        # 提取 10x10 窗口
                        window = p_data[py: py + 10, px: px + 10]

                        if window.size > 0 and not np.all(np.isnan(window)):
                            out_array[r, c] = np.nanmedian(window)

        # 3. 保存并记录结果
        out_name = f_name.replace('0.01deg.tif', '0.1deg.tif')
        save_path = os.path.join(output_dir, out_name)
        m_meta.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(save_path, 'w', **m_meta) as dst:
            dst.write(out_array, 1)
        results[out_name] = out_array
        print(f"✅ 完成: {out_name}")

    # 4. 绘图
    plot_final(results)


def plot_final(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (title, data) in enumerate(results.items()):
        # 排除无效值后计算显示范围，防止绘图全白
        valid = data[~np.isnan(data)]
        if valid.size > 0:
            im = axes[i].imshow(data, cmap='RdYlGn_r', vmin=np.nanmin(valid), vmax=np.nanmax(valid))
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(title)
    plt.show()


if __name__ == "__main__":
    smart_resample()