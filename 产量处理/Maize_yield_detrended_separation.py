# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. 输入输出路径
# =========================================================
input_file = r"C:\Users\wangrui\Desktop\作物单产2001-2022\作物单产汇总整理_2002-2022.xlsx"
output_file = r"C:\Users\wangrui\Desktop\作物单产2001-2022\玉米去趋势分离单产结果_二次拟合.xlsx"

# 图片输出总目录
plot_root = r"C:\Users\wangrui\Desktop\作物单产2001-2022\二次拟合图片_玉米"

# =========================================================
# 2. 参数设置
# =========================================================
target_sheets = [
    "玉米单位面积产量公斤公顷"
]

# True = 每张图都弹窗显示；False = 只保存不弹窗
show_plots = False

# 是否保存图片
save_plots = True

# 图片分辨率
dpi = 150

# matplotlib 中文显示设置（适合 PyCharm）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 3. 文件名清理函数
# =========================================================
def safe_filename(name):
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return str(name).strip()


# =========================================================
# 4. 二次趋势去除函数
# =========================================================
def detrend_yield_quadratic(df, crop_name="作物"):
    """
    输入:
        df: 第一列为省份，后面列为年份单产
    输出:
        actual_df   : 实际单产
        trend_df    : 趋势单产（二次拟合）
        climate_df  : 气象单产 = 实际单产 - 趋势单产
        rel_clim_df : 相对气象单产 = 气象单产 / 趋势单产 * 100
    """
    df = df.copy()

    province_col = df.columns[0]
    year_cols = list(df.columns[1:])
    years = np.array([int(str(c).replace("年", "")) for c in year_cols], dtype=float)

    actual_df = df.copy()
    trend_df = pd.DataFrame(columns=df.columns)
    climate_df = pd.DataFrame(columns=df.columns)
    rel_clim_df = pd.DataFrame(columns=df.columns)

    trend_df[province_col] = df[province_col]
    climate_df[province_col] = df[province_col]
    rel_clim_df[province_col] = df[province_col]

    print("\n" + "=" * 80)
    print(f"开始处理: {crop_name}")
    print("=" * 80)

    for idx in range(len(df)):
        province = df.loc[idx, province_col]
        y = pd.to_numeric(df.loc[idx, year_cols], errors="coerce").values.astype(float)

        valid_mask = ~np.isnan(y)

        if valid_mask.sum() < 3:
            print(f"⚠️ {province}: 有效年份不足 3 个，无法进行二次拟合，跳过")
            trend = np.full_like(y, np.nan, dtype=float)
            climate = np.full_like(y, np.nan, dtype=float)
            rel_climate = np.full_like(y, np.nan, dtype=float)
        else:
            x_valid = years[valid_mask]
            y_valid = y[valid_mask]

            # 二次拟合: y = a*x^2 + b*x + c
            coeffs = np.polyfit(x_valid, y_valid, 2)
            a, b, c = coeffs

            trend = a * years**2 + b * years + c
            climate = y - trend

            rel_climate = np.full_like(y, np.nan, dtype=float)
            nonzero_mask = (~np.isnan(trend)) & (trend != 0) & (~np.isnan(y))
            rel_climate[nonzero_mask] = climate[nonzero_mask] / trend[nonzero_mask] * 100

            print(f"✅ {province}: 二次拟合完成 (a={a:.6f}, b={b:.4f}, c={c:.2f})")

        trend_df.loc[idx, year_cols] = trend
        climate_df.loc[idx, year_cols] = climate
        rel_clim_df.loc[idx, year_cols] = rel_climate

    return actual_df, trend_df, climate_df, rel_clim_df


# =========================================================
# 5. 绘制各省“实际单产 + 二次拟合趋势单产”图
# =========================================================
def plot_actual_and_trend_all_provinces(actual_df, trend_df, crop_name,
                                        output_dir=None,
                                        show_plots=False,
                                        save_plots=True,
                                        dpi=150):
    province_col = actual_df.columns[0]
    year_cols = list(actual_df.columns[1:])
    years = np.array([int(str(c).replace("年", "")) for c in year_cols], dtype=int)

    if save_plots and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 80)
    print(f"开始绘制 {crop_name} 各省二次拟合单产图")
    print("-" * 80)

    for idx in range(len(actual_df)):
        province = actual_df.loc[idx, province_col]

        actual_y = pd.to_numeric(actual_df.loc[idx, year_cols], errors="coerce").values.astype(float)
        trend_y = pd.to_numeric(trend_df.loc[idx, year_cols], errors="coerce").values.astype(float)

        valid_actual = ~np.isnan(actual_y)
        valid_trend = ~np.isnan(trend_y)

        if valid_actual.sum() == 0:
            print(f"⚠️ {province}: 实际单产全为空，跳过绘图")
            continue

        plt.figure(figsize=(10, 6))

        # 实际单产
        plt.plot(
            years[valid_actual],
            actual_y[valid_actual],
            marker='o',
            linewidth=1.8,
            label='实际单产'
        )

        # 二次拟合趋势单产
        if valid_trend.sum() > 0:
            plt.plot(
                years[valid_trend],
                trend_y[valid_trend],
                linestyle='--',
                linewidth=2.2,
                label='二次拟合趋势单产'
            )

        plt.title(f"{crop_name} - {province} 实际单产与二次拟合趋势", fontsize=14)
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("单产（公斤/公顷）", fontsize=12)
        plt.xticks(years, rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_plots and output_dir is not None:
            file_name = f"{safe_filename(province)}_{crop_name}_二次拟合单产.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()

        print(f"🖼️ 已完成绘图: {province}")

    print(f"✅ {crop_name} 绘图完成")


# =========================================================
# 6. 绘制各省气象单产图
# =========================================================
def plot_climate_yield_all_provinces(climate_df, crop_name,
                                     output_dir=None,
                                     show_plots=False,
                                     save_plots=True,
                                     dpi=150):
    province_col = climate_df.columns[0]
    year_cols = list(climate_df.columns[1:])
    years = np.array([int(str(c).replace("年", "")) for c in year_cols], dtype=int)

    if save_plots and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 80)
    print(f"开始绘制 {crop_name} 各省气象单产图")
    print("-" * 80)

    for idx in range(len(climate_df)):
        province = climate_df.loc[idx, province_col]
        y = pd.to_numeric(climate_df.loc[idx, year_cols], errors="coerce").values.astype(float)

        valid_mask = ~np.isnan(y)
        if valid_mask.sum() == 0:
            print(f"⚠️ {province}: 气象单产全为空，跳过")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(years[valid_mask], y[valid_mask], marker='o', linewidth=1.8, label='气象单产')
        plt.axhline(0, linestyle='--', linewidth=1)

        plt.title(f"{crop_name} - {province} 气象单产（去趋势后）", fontsize=14)
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("气象单产", fontsize=12)
        plt.xticks(years, rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_plots and output_dir is not None:
            file_name = f"{safe_filename(province)}_{crop_name}_气象单产.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()

        print(f"🖼️ 已完成气象单产绘图: {province}")

    print(f"✅ {crop_name} 气象单产绘图完成")


# =========================================================
# 7. 主程序
# =========================================================
def main():
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    print("=" * 80)
    print("读取 Excel 文件")
    print(f"输入文件: {input_file}")
    print("=" * 80)

    os.makedirs(plot_root, exist_ok=True)

    xls = pd.ExcelFile(input_file)
    all_sheet_names = xls.sheet_names

    print("检测到工作表：")
    for s in all_sheet_names:
        print(" -", s)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name in target_sheets:
            if sheet_name not in all_sheet_names:
                print(f"\n⚠️ 未找到工作表: {sheet_name}，跳过")
                continue

            df = pd.read_excel(input_file, sheet_name=sheet_name)
            df = df.dropna(how="all").reset_index(drop=True)

            # 去趋势分离
            actual_df, trend_df, climate_df, rel_clim_df = detrend_yield_quadratic(
                df, crop_name=sheet_name
            )

            short_name = "玉米"

            # 保存 Excel 结果
            actual_df.to_excel(writer, sheet_name=f"{short_name}_实际单产", index=False)
            trend_df.to_excel(writer, sheet_name=f"{short_name}_趋势单产_二次", index=False)
            climate_df.to_excel(writer, sheet_name=f"{short_name}_气象单产_二次", index=False)
            rel_clim_df.to_excel(writer, sheet_name=f"{short_name}_相对气象单产%_二次", index=False)

            # 图片目录
            crop_plot_dir = os.path.join(plot_root, short_name, "实际单产_趋势单产")
            climate_plot_dir = os.path.join(plot_root, short_name, "气象单产")

            # 1）绘制实际单产 + 趋势单产图
            plot_actual_and_trend_all_provinces(
                actual_df=actual_df,
                trend_df=trend_df,
                crop_name=short_name,
                output_dir=crop_plot_dir,
                show_plots=show_plots,
                save_plots=save_plots,
                dpi=dpi
            )

            # 2）绘制气象单产图
            plot_climate_yield_all_provinces(
                climate_df=climate_df,
                crop_name=short_name,
                output_dir=climate_plot_dir,
                show_plots=False,
                save_plots=True,
                dpi=dpi
            )

    print("\n" + "=" * 80)
    print("🎉 处理完成")
    print(f"Excel结果已保存到: {output_file}")
    print(f"图片已保存到: {plot_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()