# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. 文件路径
# =========================================================
r2_file = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产\两阶段R2汇总结果_气象单产.xlsx"
sheet_name = "All_R2_Results"

output_dir = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产\SCI分面图"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. 全局绘图参数
# =========================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 统一字号
TITLE_SIZE = 16
SUBTITLE_SIZE = 12
LABEL_SIZE = 12
TICK_SIZE = 10

# 指标和尺度顺序
index_order = ["SPI", "SPEI", "SVPD", "SEDI", "SPET"]
scale_order = ["1month", "3month", "6month", "9month", "12month"]
scale_labels = ["1", "3", "6", "9", "12"]
stage_order = ["Stage 1", "Stage 2"]

crop_map = {
    "Winter_Wheat": "Winter wheat",
    "Spring_Wheat": "Spring wheat"
}

# 原始中文阶段 -> 英文阶段
stage_map = {
    "阶段1": "Stage 1",
    "阶段2": "Stage 2"
}

# =========================================================
# 3. 读取数据
# =========================================================
def load_r2_data(file_path, sheet_name="All_R2_Results"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(how="all").reset_index(drop=True)

    required_cols = ["阶段", "作物", "省份", "指数", "尺度", "R2"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要字段: {c}")

    df = df.copy()
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df = df.dropna(subset=["R2"]).reset_index(drop=True)

    # 作物英文名
    df["Crop_EN"] = df["作物"].map(crop_map)

    # 阶段英文名
    df["Stage_EN"] = df["阶段"].map(stage_map)

    df["指数"] = pd.Categorical(df["指数"], categories=index_order, ordered=True)
    df["尺度"] = pd.Categorical(df["尺度"], categories=scale_order, ordered=True)
    df["Stage_EN"] = pd.Categorical(df["Stage_EN"], categories=stage_order, ordered=True)

    df = df.sort_values(["Crop_EN", "Stage_EN", "指数", "尺度", "省份"]).reset_index(drop=True)
    return df

# =========================================================
# 4. 单个作物的 2×5 分面箱线图
# =========================================================
def plot_crop_facet_boxplot(df, crop_code, crop_en, output_dir):
    df_crop = df[df["作物"] == crop_code].copy()
    if df_crop.empty:
        print(f"⚠ {crop_en} 无数据")
        return

    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(18, 8), sharey=True
    )

    y_max = df_crop["R2"].max()
    if pd.isna(y_max):
        y_max = 1.0
    y_upper = min(1.0, max(0.1, y_max * 1.1))

    for i, stage in enumerate(stage_order):
        for j, idx in enumerate(index_order):
            ax = axes[i, j]

            sub = df_crop[(df_crop["Stage_EN"] == stage) & (df_crop["指数"] == idx)].copy()

            data_list = []
            valid_labels = []

            for sc, sc_label in zip(scale_order, scale_labels):
                vals = sub[sub["尺度"] == sc]["R2"].dropna().values
                if len(vals) > 0:
                    data_list.append(vals)
                else:
                    data_list.append(np.array([np.nan]))
                valid_labels.append(sc_label)

            bp = ax.boxplot(
                data_list,
                patch_artist=True,
                widths=0.55,
                showmeans=False,
                showfliers=True

            )

            # SCI 风格：白底黑框，红色中位线
            for box in bp["boxes"]:
                box.set(facecolor="white", edgecolor="black", linewidth=1.0)
            for whisker in bp["whiskers"]:
                whisker.set(color="black", linewidth=1.0)
            for cap in bp["caps"]:
                cap.set(color="black", linewidth=1.0)
            for median in bp["medians"]:
                median.set(color="red", linewidth=1.4)
            for flier in bp["fliers"]:
                flier.set(
                    marker="o",
                    markerfacecolor="gray",
                    markeredgecolor="gray",
                    markersize=3,
                    alpha=0.5
                )

            ax.set_xticks(range(1, len(valid_labels) + 1))
            ax.set_xticklabels(valid_labels, fontsize=TICK_SIZE)
            ax.tick_params(axis="y", labelsize=TICK_SIZE)

            ax.set_ylim(0, y_upper)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

            # 上排标题
            if i == 0:
                ax.set_title(idx, fontsize=SUBTITLE_SIZE, pad=8)

            # 左列标阶段
            if j == 0:
                ax.set_ylabel(f"{stage}\n$R^2$", fontsize=LABEL_SIZE)
            else:
                ax.set_ylabel("")

            # 下排显示 x 轴标题
            if i == 1:
                ax.set_xlabel("Time scale (month)", fontsize=LABEL_SIZE)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    fig.suptitle(
        f"{crop_en}: provincial $R^2$ distributions of drought indices",
        fontsize=TITLE_SIZE,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(output_dir, f"{crop_en}_5indices_5scales_boxplot.png")
    out_pdf = os.path.join(output_dir, f"{crop_en}_5indices_5scales_boxplot.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"✅ 已保存: {out_png}")
    print(f"✅ 已保存: {out_pdf}")

# =========================================================
# 5. 主程序
# =========================================================
def main():
    print("=" * 80)
    print("开始绘制：5 indices × 5 scales boxplots (SCI style)")
    print("=" * 80)

    df = load_r2_data(r2_file, sheet_name=sheet_name)
    print(f"✅ R2 数据读取完成，记录数: {len(df)}")

    for crop_code, crop_en in crop_map.items():
        print(f"\n正在绘制：{crop_en}")
        plot_crop_facet_boxplot(df, crop_code, crop_en, output_dir)

    print("\n" + "=" * 80)
    print("🎉 全部绘制完成")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()