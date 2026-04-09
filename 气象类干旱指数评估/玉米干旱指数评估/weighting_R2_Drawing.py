# -*- coding: utf-8 -*-
import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# =========================================================
# 0. matplotlib 设置
# =========================================================
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 输入文件路径
# =========================================================
# 前面计算好的 玉米 R2 总结果表
r2_file = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产_玉米\玉米_两阶段R2汇总结果_气象单产.xlsx"

# 改成：春夏玉米产量表（用于算权重）
production_file = r"C:\Users\wangrui\Desktop\作物产量\整理后的作物产量2002-2022.xlsx"

# 输出目录
output_dir = r"E:\data\Province_Stage_Index_Mean\最终筛选结果"
os.makedirs(output_dir, exist_ok=True)

# 统计结果输出
output_excel = os.path.join(output_dir, "春夏玉米_全国加权平均R2最优指标_按多年平均产量权重.xlsx")

# 热力图输出
heatmap_png = os.path.join(output_dir, "春夏玉米_分阶段各组合全国加权R2热力图_指标×尺度.png")
heatmap_pdf = os.path.join(output_dir, "春夏玉米_分阶段各组合全国加权R2热力图_指标×尺度.pdf")


# =========================================================
# 2. 基础参数
# =========================================================
YEARS = list(range(2002, 2023))

MAIZE_CROPS = ["Spring_Maize", "Summer_Maize"]

# 作物对应的产量sheet
PRODUCTION_SHEET_MAP = {
    "Spring_Maize": "春玉米产量（万吨）",
    "Summer_Maize": "夏玉米产量（万吨）"
}

crop_en_map = {
    "Spring_Maize": "Spring Maize",
    "Summer_Maize": "Summer Maize"
}

index_order = ["SPI", "SPEI", "SPET", "SVPD", "SEDI", "SSI", "VCI", "TCI", "VHI"]
scale_order = ["1", "3", "6", "9", "12"]
stage_order = ["阶段1", "阶段2", "全阶段", "Stage1", "Stage2", "Stage1+2"]

annotate = True
vmin = 0.0
vmax = 1.0


# =========================================================
# 3. 工具函数
# =========================================================
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")


def normalize_province_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()

    replace_dict = {
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
        "西藏自治区": "西藏",
        "北京市": "北京",
        "天津市": "天津",
        "上海市": "上海",
        "重庆市": "重庆",
        "香港特别行政区": "香港",
        "澳门特别行政区": "澳门",
        "辽宁省": "辽宁",
        "吉林省": "吉林",
        "黑龙江省": "黑龙江",
        "河北省": "河北",
        "山西省": "山西",
        "云南省": "云南",
        "四川省": "四川",
        "陕西省": "陕西",
        "甘肃省": "甘肃",
        "湖北省": "湖北",
        "湖南省": "湖南",
        "江西省": "江西",
        "青海省": "青海",
        "山东省": "山东",
        "河南省": "河南",
        "安徽省": "安徽",
        "贵州省": "贵州",
        "江苏省": "江苏",
        "浙江省": "浙江",
        "内蒙": "内蒙古",
    }
    name = replace_dict.get(name, name)

    suffixes = ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区", "特别行政区"]
    for s in suffixes:
        if name.endswith(s):
            name = name.replace(s, "")

    return name.strip()


def province_key(name):
    name = normalize_province_name(name)
    if name is None or len(name) == 0:
        return None
    return name[:2]


def extract_scale_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1)
    return s


def sort_by_custom_order(values, custom_order):
    order_map = {v: i for i, v in enumerate(custom_order)}
    return sorted(values, key=lambda x: order_map.get(x, 9999))


# =========================================================
# 4. 读取 R2 结果表
# =========================================================
def load_r2_result(r2_excel):
    check_file_exists(r2_excel)

    df = pd.read_excel(r2_excel, sheet_name="All_R2_Results")
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["阶段", "作物", "省份", "指数", "尺度", "R2"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"R2结果表缺少必要字段: {c}")

    df = df.copy()
    df["省份"] = df["省份"].apply(normalize_province_name)
    df["省份key"] = df["省份"].apply(province_key)
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df["尺度"] = df["尺度"].apply(extract_scale_num)

    df = df.dropna(subset=["R2", "省份", "省份key", "作物", "阶段", "指数", "尺度"]).reset_index(drop=True)
    return df


# =========================================================
# 5. 从春/夏玉米产量表计算“多年平均产量权重”
# =========================================================
def build_crop_production_weight_table(prod_excel, crop_name):
    """
    对指定作物（Spring_Maize / Summer_Maize）读取对应产量sheet，
    计算：
        某省多年平均产量 / 该作物所有省份多年平均产量总和
    """
    check_file_exists(prod_excel)

    sheet_name = PRODUCTION_SHEET_MAP[crop_name]
    df = pd.read_excel(prod_excel, sheet_name=sheet_name)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    # 第一列有时叫“省份”，但实际上存的是年份，需要统一改成“年份”
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "年份"})

    if "年份" not in df.columns:
        raise ValueError(f"{sheet_name} 缺少年份列")

    province_cols = [c for c in df.columns if c != "年份"]
    if len(province_cols) == 0:
        raise ValueError(f"{sheet_name} 未识别到省份列")

    df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
    df = df[df["年份"].isin(YEARS)].copy()

    df_long = df.melt(
        id_vars=["年份"],
        value_vars=province_cols,
        var_name="省份",
        value_name="产量"
    )

    df_long["省份"] = df_long["省份"].apply(normalize_province_name)
    df_long["省份key"] = df_long["省份"].apply(province_key)
    df_long["产量"] = pd.to_numeric(df_long["产量"], errors="coerce")

    df_long = df_long.dropna(subset=["年份", "省份", "省份key", "产量"]).copy()

    avg_prod = (
        df_long.groupby(["省份key", "省份"], as_index=False)["产量"]
        .mean()
        .rename(columns={"产量": "多年平均产量"})
    )

    total_prod = avg_prod["多年平均产量"].sum()
    if total_prod == 0 or pd.isna(total_prod):
        raise ValueError(f"{sheet_name} 多年平均产量总和为0，无法计算权重")

    avg_prod["产量权重"] = avg_prod["多年平均产量"] / total_prod
    avg_prod["作物"] = crop_name

    avg_prod = avg_prod.sort_values("产量权重", ascending=False).reset_index(drop=True)
    return avg_prod, df_long


def build_all_maize_production_weights(prod_excel):
    """
    合并春玉米、夏玉米两个作物的产量权重表
    """
    all_weight = []
    all_long = []

    for crop in MAIZE_CROPS:
        weight_df, long_df = build_crop_production_weight_table(prod_excel, crop)
        all_weight.append(weight_df)
        long_df["作物"] = crop
        all_long.append(long_df)

    weight_all = pd.concat(all_weight, ignore_index=True)
    prod_long_all = pd.concat(all_long, ignore_index=True)
    return weight_all, prod_long_all


# =========================================================
# 6. 合并 R2 与产量权重
# =========================================================
def merge_r2_with_weight(df_r2, prod_excel):
    """
    将省级R2结果与春/夏玉米多年平均产量权重合并
    注意：这里必须按 作物 + 省份key 合并
    """
    weight_df, _ = build_all_maize_production_weights(prod_excel)

    merged = pd.merge(
        df_r2,
        weight_df[["作物", "省份key", "省份", "多年平均产量", "产量权重"]],
        on=["作物", "省份key"],
        how="left",
        suffixes=("", "_权重表")
    )

    merged["省份"] = merged["省份"].fillna(merged["省份_权重表"])
    if "省份_权重表" in merged.columns:
        merged = merged.drop(columns=["省份_权重表"])

    unmatched = merged[merged["产量权重"].isna()][["作物", "省份"]].drop_duplicates()
    if len(unmatched) > 0:
        print("\n⚠ 以下 作物-省份 未匹配到产量权重：")
        print(unmatched)

    merged = merged.dropna(subset=["R2", "产量权重"]).copy()

    return merged, weight_df


# =========================================================
# 7. 计算全国加权平均 R2
# =========================================================
def build_national_weighted_r2_table(merged_df):
    """
    对每个 作物 + 阶段 + 指数 + 尺度
    计算全国尺度的加权平均R2
    """
    if len(merged_df) == 0:
        return pd.DataFrame()

    def weighted_mean_r2(group):
        w = pd.to_numeric(group["产量权重"], errors="coerce")
        r2 = pd.to_numeric(group["R2"], errors="coerce")

        valid = (~w.isna()) & (~r2.isna())
        w = w[valid]
        r2 = r2[valid]

        if len(w) == 0 or w.sum() == 0:
            return np.nan

        return np.sum(w * r2) / np.sum(w)

    result = (
        merged_df.groupby(["作物", "阶段", "指数", "尺度"], as_index=False)
        .apply(lambda g: pd.Series({
            "参与省份数": g["省份key"].nunique(),
            "全国加权平均R2": weighted_mean_r2(g),
            "支持省份": "、".join(sorted(g["省份"].dropna().unique().tolist()))
        }))
        .reset_index(drop=True)
    )

    result["全国加权平均R2排名"] = (
        result.groupby(["作物", "阶段"])["全国加权平均R2"]
        .rank(ascending=False, method="min")
    )

    result = result.sort_values(
        ["作物", "阶段", "全国加权平均R2"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return result


# =========================================================
# 8. 提取全国最优指标
# =========================================================
def build_best_indicator_table(weighted_r2_df):
    if len(weighted_r2_df) == 0:
        return pd.DataFrame()

    best_df = (
        weighted_r2_df.sort_values(
            ["作物", "阶段", "全国加权平均R2"],
            ascending=[True, True, False]
        )
        .groupby(["作物", "阶段"], as_index=False)
        .first()
        .rename(columns={
            "指数": "全国最优指标",
            "尺度": "全国最优尺度",
            "全国加权平均R2": "最优指标全国加权平均R2",
            "全国加权平均R2排名": "最优指标排名"
        })
    )

    return best_df


# =========================================================
# 9. 绘制热力图
# =========================================================
def draw_weighted_r2_heatmap(weighted_r2_df, output_png, output_pdf):
    if len(weighted_r2_df) == 0:
        print("⚠ 没有可用于绘图的全国加权R2数据。")
        return

    stage_en_map = {
        "阶段1": "Stage 1",
        "阶段2": "Stage 2",
        "全阶段": "Full Stage",
        "Stage1": "Stage 1",
        "Stage2": "Stage 2",
        "Stage1+2": "Full Stage"
    }

    crops = sort_by_custom_order(
        weighted_r2_df["作物"].dropna().unique().tolist(),
        MAIZE_CROPS
    )

    stages = sort_by_custom_order(
        weighted_r2_df["阶段"].dropna().unique().tolist(),
        stage_order
    )

    combos = []
    for crop in crops:
        for stage in stages:
            sub = weighted_r2_df[
                (weighted_r2_df["作物"] == crop) &
                (weighted_r2_df["阶段"] == stage)
            ]
            if len(sub) > 0:
                combos.append((crop, stage))

    if len(combos) == 0:
        print("⚠ 未找到可绘制的 作物 × 阶段 组合。")
        return

    n = len(combos)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5.5 * nrows),
        constrained_layout=True
    )

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    cmap = plt.cm.Reds
    im = None

    for ax, (crop, stage) in zip(axes, combos):
        sub = weighted_r2_df[
            (weighted_r2_df["作物"] == crop) &
            (weighted_r2_df["阶段"] == stage)
        ].copy()

        sub["尺度"] = sub["尺度"].astype(str)

        row_vals = sub["指数"].dropna().unique().tolist()
        col_vals = sub["尺度"].dropna().unique().tolist()

        row_sorted = [x for x in index_order if x in row_vals] + [x for x in row_vals if x not in index_order]
        col_sorted = [x for x in scale_order if x in col_vals] + [x for x in col_vals if x not in scale_order]

        pivot = sub.pivot(index="指数", columns="尺度", values="全国加权平均R2")
        pivot = pivot.reindex(index=row_sorted, columns=col_sorted)

        data = pivot.values.astype(float)

        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto"
        )

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(
            [f"{c}-month" if str(c).isdigit() else str(c) for c in pivot.columns],
            fontsize=11
        )
        ax.set_yticklabels(pivot.index, fontsize=11)

        crop_en = crop_en_map.get(crop, str(crop))
        stage_en = stage_en_map.get(stage, str(stage))

        ax.set_title(f"{crop_en} - {stage_en}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Scale", fontsize=12)
        ax.set_ylabel("Index", fontsize=12)

        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        if annotate:
            threshold = (vmin + vmax) / 2
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if np.isnan(val):
                        continue
                    txt_color = "black" if val < threshold else "white"
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=10, color=txt_color
                    )

        if np.isfinite(data).any():
            max_val = np.nanmax(data)
            max_positions = np.argwhere(data == max_val)

            for pos in max_positions:
                i, j = pos
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5
                )
                ax.add_patch(rect)

    for k in range(len(combos), len(axes)):
        fig.delaxes(axes[k])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:len(combos)], shrink=0.92, pad=0.02)
        cbar.set_label("National Weighted R²", fontsize=12)

    fig.suptitle(
        "Stage-wise National Weighted R² Heatmaps for Maize",
        fontsize=18,
        fontweight="bold"
    )

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Heatmap saved: {output_png}")
    print(f"✅ Heatmap saved: {output_pdf}")


# =========================================================
# 10. 主程序
# =========================================================
def main():
    print("=" * 100)
    print("开始生成：春夏玉米 各指标全国尺度多年平均产量加权R2 + 热力图")
    print("=" * 100)

    # 1) 读取各省各指标R2结果
    df_r2 = load_r2_result(r2_file)
    print(f"✅ R2结果读取完成，记录数: {len(df_r2)}")

    # 2) 与春/夏玉米多年平均产量权重合并
    df_merge, df_weight = merge_r2_with_weight(df_r2, production_file)
    print("✅ 各省指标R2与春/夏玉米多年平均产量权重合并完成")

    # 3) 计算全国加权平均R2
    df_weighted_r2 = build_national_weighted_r2_table(df_merge)
    print("✅ 全国尺度加权平均R2计算完成")

    # 4) 提取全国最优指标
    df_best = build_best_indicator_table(df_weighted_r2)
    print("✅ 全国最优指标提取完成")

    # 5) 输出 Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        if len(df_r2) > 0:
            df_r2.to_excel(writer, sheet_name="各省各指标R2明细", index=False)

        if len(df_weight) > 0:
            df_weight.to_excel(writer, sheet_name="各省多年平均产量权重", index=False)

        if len(df_merge) > 0:
            df_merge.to_excel(writer, sheet_name="R2值_权重合并明细", index=False)

        if len(df_weighted_r2) > 0:
            df_weighted_r2.to_excel(writer, sheet_name="各指标全国加权平均R2排名", index=False)

        if len(df_best) > 0:
            df_best.to_excel(writer, sheet_name="全国最优指标汇总", index=False)

        for crop in MAIZE_CROPS:
            for stage in ["阶段1", "阶段2", "全阶段", "Stage1", "Stage2", "Stage1+2"]:
                sub = df_weighted_r2[
                    (df_weighted_r2["作物"] == crop) &
                    (df_weighted_r2["阶段"] == stage)
                ].copy()

                if len(sub) > 0:
                    sheet_name = f"{crop}_{stage}"[:31]
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Excel结果已保存: {output_excel}")

    # 6) 绘制热力图
    draw_weighted_r2_heatmap(
        weighted_r2_df=df_weighted_r2,
        output_png=heatmap_png,
        output_pdf=heatmap_pdf
    )

    print("\n" + "=" * 100)
    print("🎉 全部完成")
    print("=" * 100)


if __name__ == "__main__":
    main()