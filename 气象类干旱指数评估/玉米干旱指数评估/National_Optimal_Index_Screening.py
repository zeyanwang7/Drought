# -*- coding: utf-8 -*-
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# 1. 输入文件路径
# =========================================================
# 你前面已经生成好的 玉米两阶段 R2 总结果表
r2_file = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产_玉米\玉米_两阶段R2汇总结果_气象单产.xlsx"

# 你前面计算 R2 时使用的统一玉米气象单产文件
yield_file = r"C:\Users\wangrui\Desktop\作物单产2001-2022\玉米去趋势分离单产结果_二次拟合.xlsx"

# 输出目录
output_dir = r"E:\data\Province_Stage_Index_Mean\最终筛选结果"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "春夏玉米_全国加权平均R2最优指标_按统一玉米单产加权.xlsx")


# =========================================================
# 2. 基础参数
# =========================================================
YEARS = list(range(2002, 2023))

# 你的统一玉米气象单产sheet名称（与R2代码保持一致）
YIELD_SHEET_NAME = "玉米_气象单产_二次"

# 你的玉米作物sheet / 作物代码（与R2代码保持一致）
MAIZE_CROPS = ["Spring_Maize", "Summer_Maize"]

# 作物中英文映射
crop_cn_map = {
    "Spring_Maize": "Spring_Maize",
    "Summer_Maize": "Summer_Maize"
}


# =========================================================
# 3. 工具函数
# =========================================================
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")


def normalize_province_name(name):
    """
    与你R2代码保持一致：统一省份名称
    """
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
        "内蒙": "内蒙古",
    }
    name = replace_dict.get(name, name)

    suffixes = ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区", "特别行政区"]
    for s in suffixes:
        if name.endswith(s):
            name = name.replace(s, "")

    return name.strip()


def province_key(name):
    """
    与你R2代码保持一致：前两个字模糊匹配key
    """
    name = normalize_province_name(name)
    if name is None or len(name) == 0:
        return None
    return name[:2]


def detect_year_columns_from_yield(df):
    """
    识别玉米单产表中的年份列：如 2002年, 2003年, ...
    """
    year_cols = [f"{y}年" for y in YEARS if f"{y}年" in df.columns]
    if len(year_cols) == 0:
        raise ValueError(f"{YIELD_SHEET_NAME} 未识别到年份列（如 2002年）")
    return year_cols


# =========================================================
# 4. 读取 R2 结果表
# =========================================================
def load_r2_result(r2_excel):
    """
    读取前面生成的两阶段R2总表
    预期 sheet: All_R2_Results
    预期字段:
    阶段, 作物, 省份, 省份key, 指数, 尺度, 有效年份数, R2, 回归系数, 截距
    """
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

    df = df.dropna(subset=["R2", "省份", "省份key", "作物"]).reset_index(drop=True)
    return df


# =========================================================
# 5. 读取统一玉米气象单产表并计算权重
# =========================================================
def build_maize_yield_weight_table(yield_excel):
    """
    从统一玉米气象单产表中读取省级单产，
    计算 2002-2022 多年平均单产和单产权重

    注意：
    这里不区分 Spring_Maize / Summer_Maize，
    因为你前面的R2代码就是统一使用同一张玉米单产表。
    """
    check_file_exists(yield_excel)

    df = pd.read_excel(yield_excel, sheet_name=YIELD_SHEET_NAME)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    if "省份" not in df.columns:
        raise ValueError(f"{YIELD_SHEET_NAME} 缺少“省份”列")

    year_cols = detect_year_columns_from_yield(df)

    df["省份"] = df["省份"].astype(str).str.strip()
    df["省份_norm"] = df["省份"].apply(normalize_province_name)
    df["省份key"] = df["省份"].apply(province_key)

    df_long = df.melt(
        id_vars=["省份", "省份_norm", "省份key"],
        value_vars=year_cols,
        var_name="年份",
        value_name="单产"
    )

    df_long["年份"] = df_long["年份"].str.replace("年", "", regex=False)
    df_long["年份"] = pd.to_numeric(df_long["年份"], errors="coerce")
    df_long["单产"] = pd.to_numeric(df_long["单产"], errors="coerce")

    df_long = df_long.dropna(subset=["省份key", "年份", "单产"]).copy()
    df_long["年份"] = df_long["年份"].astype(int)
    df_long = df_long[df_long["年份"].isin(YEARS)].copy()

    avg_yield = (
        df_long.groupby(["省份key", "省份_norm"], as_index=False)["单产"]
        .mean()
        .rename(columns={
            "省份_norm": "省份",
            "单产": "多年平均单产"
        })
    )

    total = avg_yield["多年平均单产"].sum()
    if total == 0 or pd.isna(total):
        raise ValueError("玉米多年平均单产总和为0，无法计算权重")

    avg_yield["单产权重"] = avg_yield["多年平均单产"] / total
    avg_yield = avg_yield.sort_values("单产权重", ascending=False).reset_index(drop=True)

    return avg_yield, df_long


# =========================================================
# 6. 合并 R2 与单产权重
# =========================================================
def merge_r2_with_weight(df_r2, yield_excel):
    """
    将省级R2结果与统一玉米单产权重合并
    """
    weight_df, yield_long_df = build_maize_yield_weight_table(yield_excel)

    merged = pd.merge(
        df_r2,
        weight_df[["省份key", "省份", "多年平均单产", "单产权重"]],
        on="省份key",
        how="left",
        suffixes=("", "_权重表")
    )

    # 优先保留R2结果里的省份名称；如果缺失则用权重表中的
    merged["省份"] = merged["省份"].fillna(merged["省份_权重表"])
    if "省份_权重表" in merged.columns:
        merged = merged.drop(columns=["省份_权重表"])

    unmatched = merged[merged["单产权重"].isna()]["省份"].drop_duplicates().tolist()
    if unmatched:
        print("\n⚠ 以下省份未匹配到玉米单产权重：")
        for p in unmatched:
            print("   -", p)

    # 以单产表为标准：未匹配到单产权重的省份不参与全国加权
    merged = merged.dropna(subset=["R2", "单产权重"]).copy()

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
        w = pd.to_numeric(group["单产权重"], errors="coerce")
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
    """
    每个 作物 + 阶段，选全国加权平均R2最大的 指数+尺度
    """
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
# 9. 主程序
# =========================================================
def main():
    print("=" * 100)
    print("开始生成：春夏玉米 各指标全国尺度单产加权平均R2 + 全国最优指标")
    print("=" * 100)

    # 1) 读取各省各指标R2结果
    df_r2 = load_r2_result(r2_file)
    print(f"✅ R2结果读取完成，记录数: {len(df_r2)}")

    # 2) 与统一玉米单产权重合并
    df_merge, df_weight = merge_r2_with_weight(df_r2, yield_file)
    print("✅ 各省指标R2与统一玉米单产权重合并完成")

    # 3) 计算全国加权平均R2
    df_weighted_r2 = build_national_weighted_r2_table(df_merge)
    print("✅ 全国尺度加权平均R2计算完成")

    # 4) 提取全国最优指标
    df_best = build_best_indicator_table(df_weighted_r2)
    print("✅ 全国最优指标提取完成")

    # =====================================================
    # 5. 输出 Excel
    # =====================================================
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        if len(df_r2) > 0:
            df_r2.to_excel(writer, sheet_name="各省各指标R2明细", index=False)

        if len(df_weight) > 0:
            df_weight.to_excel(writer, sheet_name="各省多年平均单产权重", index=False)

        if len(df_merge) > 0:
            df_merge.to_excel(writer, sheet_name="R2值_权重合并明细", index=False)

        if len(df_weighted_r2) > 0:
            df_weighted_r2.to_excel(writer, sheet_name="各指标全国加权平均R2排名", index=False)

        if len(df_best) > 0:
            df_best.to_excel(writer, sheet_name="全国最优指标汇总", index=False)

        # 分作物分阶段输出
        for crop in MAIZE_CROPS:
            for stage in ["阶段1", "阶段2"]:
                sub = df_weighted_r2[
                    (df_weighted_r2["作物"] == crop) &
                    (df_weighted_r2["阶段"] == stage)
                ].copy()

                if len(sub) > 0:
                    sheet_name = f"{crop}_{stage}"[:31]
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n" + "=" * 100)
    print("🎉 结果已生成")
    print(f"输出文件: {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()