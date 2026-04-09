# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. 文件路径
# =========================================================
stage1_file = r"E:\data\Province_Stage_Index_Mean\整理后\小麦阶段1干旱指数_宽表_2002_2022.xlsx"
stage2_file = r"E:\data\Province_Stage_Index_Mean\整理后\小麦阶段2干旱指数_宽表_2002_2022.xlsx"

yield_file = r"C:\Users\wangrui\Desktop\作物单产2001-2022\冬春小麦去趋势分离单产结果_二次拟合.xlsx"

output_dir = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. 基础参数
# =========================================================
YEARS = list(range(2002, 2023))   # 2002-2022
MIN_VALID_SAMPLES = 6             # 至少6个有效年份才计算R2

crop_sheet_mapping = {
    "Winter_Wheat": "冬小麦_气象单产_二次",
    "Spring_Wheat": "春小麦_气象单产_二次",
}

# =========================================================
# 3. 工具函数
# =========================================================
def normalize_province_name(name):
    """统一省份名称"""
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
    """前两个字模糊匹配key"""
    name = normalize_province_name(name)
    if name is None or len(name) == 0:
        return None
    return name[:2]


def parse_indicator_col(col_name):
    """
    解析指标列名，如：
    SPI_1month
    SPEI_3month
    SPET_12month
    """
    col_name = str(col_name).strip()
    m = re.match(r"^([A-Za-z]+)_(\d+month)$", col_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def load_yield_long(yield_excel):
    """
    读取气象单产表，转成长表：
    省份 | 省份key | 作物 | 年份 | 单产
    """
    all_list = []

    for crop_name, sheet_name in crop_sheet_mapping.items():
        df = pd.read_excel(yield_excel, sheet_name=sheet_name)
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        if "省份" not in df.columns:
            raise ValueError(f"{sheet_name} 缺少“省份”列")

        year_cols = [f"{y}年" for y in YEARS if f"{y}年" in df.columns]
        if len(year_cols) == 0:
            raise ValueError(f"{sheet_name} 未识别到年份列")

        df["省份"] = df["省份"].astype(str).str.strip()
        df["省份key"] = df["省份"].apply(province_key)
        df["作物"] = crop_name

        df_long = df.melt(
            id_vars=["省份", "省份key", "作物"],
            value_vars=year_cols,
            var_name="年份",
            value_name="单产"
        )

        df_long["年份"] = df_long["年份"].str.replace("年", "", regex=False).astype(int)
        df_long["单产"] = pd.to_numeric(df_long["单产"], errors="coerce")

        all_list.append(df_long)

    out = pd.concat(all_list, ignore_index=True)
    out = out.dropna(subset=["省份key", "年份", "单产"]).reset_index(drop=True)
    return out


def load_stage_wide(stage_excel):
    """
    读取阶段宽表（每个文件有两个sheet：Spring_Wheat, Winter_Wheat）
    转成长表：
    年份 | 省份 | 省份key | 作物 | 指标列名 | 指数值 | 指数 | 尺度
    """
    all_list = []

    xls = pd.ExcelFile(stage_excel)
    for crop_name in ["Spring_Wheat", "Winter_Wheat"]:
        if crop_name not in xls.sheet_names:
            print(f"⚠ 未找到工作表: {crop_name}")
            continue

        df = pd.read_excel(stage_excel, sheet_name=crop_name)
        df.columns = [str(c).strip() for c in df.columns]

        required_cols = ["年份", "省份", "作物"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"{stage_excel} - {crop_name} 缺少必要列: {c}")

        df["省份"] = df["省份"].astype(str).str.strip()
        df["省份key"] = df["省份"].apply(province_key)

        meta_cols = {"年份", "省份", "作物", "阶段"}
        indicator_cols = [c for c in df.columns if c not in meta_cols]

        valid_indicator_cols = []
        for c in indicator_cols:
            idx_name, scale_name = parse_indicator_col(c)
            if idx_name is not None:
                valid_indicator_cols.append(c)

        if len(valid_indicator_cols) == 0:
            raise ValueError(f"{stage_excel} - {crop_name} 没有识别到指标列")

        df_long = df.melt(
            id_vars=["年份", "省份", "省份key", "作物"],
            value_vars=valid_indicator_cols,
            var_name="指标列名",
            value_name="指数值"
        )

        df_long["年份"] = pd.to_numeric(df_long["年份"], errors="coerce").astype("Int64")
        df_long["指数值"] = pd.to_numeric(df_long["指数值"], errors="coerce")
        df_long = df_long.dropna(subset=["年份", "省份key"]).reset_index(drop=True)
        df_long["年份"] = df_long["年份"].astype(int)

        df_long[["指数", "尺度"]] = df_long["指标列名"].apply(
            lambda x: pd.Series(parse_indicator_col(x))
        )

        all_list.append(df_long)

    out = pd.concat(all_list, ignore_index=True)
    out = out.dropna(subset=["指数", "尺度"]).reset_index(drop=True)
    return out


def calc_r2_for_group(df_group):
    """
    对某个 省-作物-指标-尺度 计算R²
    回归形式：单产 ~ 干旱指数
    """
    sub = df_group[["指数值", "单产", "年份"]].dropna().copy()
    sub = sub[np.isfinite(sub["指数值"]) & np.isfinite(sub["单产"])]

    n = len(sub)
    if n < MIN_VALID_SAMPLES:
        return pd.Series({
            "有效年份数": n,
            "R2": np.nan,
            "回归系数": np.nan,
            "截距": np.nan
        })

    if sub["指数值"].nunique() < 2 or sub["单产"].nunique() < 2:
        return pd.Series({
            "有效年份数": n,
            "R2": np.nan,
            "回归系数": np.nan,
            "截距": np.nan
        })

    X = sub["指数值"].values.reshape(-1, 1)
    y = sub["单产"].values

    model = LinearRegression()
    model.fit(X, y)

    return pd.Series({
        "有效年份数": n,
        "R2": model.score(X, y),
        "回归系数": model.coef_[0],
        "截距": model.intercept_
    })


def compute_stage_r2(stage_excel, yield_long_df, stage_name):
    """计算某一阶段R²"""
    print("\n" + "=" * 90)
    print(f"开始计算 {stage_name} 的 R2")
    print("=" * 90)

    stage_long_df = load_stage_wide(stage_excel)

    merged = pd.merge(
        stage_long_df,
        yield_long_df[["省份key", "作物", "年份", "单产"]],
        on=["省份key", "作物", "年份"],
        how="inner"
    )

    merged = merged.sort_values(["作物", "省份", "指数", "尺度", "年份"]).reset_index(drop=True)

    print(f"合并后记录数: {len(merged)}")
    print(f"作物: {merged['作物'].dropna().unique().tolist()}")
    print(f"指数: {merged['指数'].dropna().unique().tolist()}")

    result = (
        merged.groupby(["作物", "省份key", "指数", "尺度"], dropna=False)
        .apply(calc_r2_for_group)
        .reset_index()
    )

    province_name_map = (
        merged.groupby("省份key")["省份"]
        .agg(lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else None)
        .reset_index()
        .rename(columns={"省份": "省份"})
    )

    result = pd.merge(result, province_name_map, on="省份key", how="left")

    result["阶段"] = stage_name
    result = result[[
        "阶段", "作物", "省份", "省份key", "指数", "尺度",
        "有效年份数", "R2", "回归系数", "截距"
    ]]

    result = result.sort_values(["作物", "省份", "指数", "尺度"]).reset_index(drop=True)
    return result, merged


def build_best_indicator_table(result_df):
    """每个省、每个作物、每个阶段，选R²最高的指标"""
    sub = result_df.dropna(subset=["R2"]).copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["阶段", "作物", "省份", "最优指标", "最优尺度", "最大R2"])

    idx = sub.groupby(["阶段", "作物", "省份"])["R2"].idxmax()
    best = sub.loc[idx, ["阶段", "作物", "省份", "指数", "尺度", "R2"]].copy()
    best = best.rename(columns={
        "指数": "最优指标",
        "尺度": "最优尺度",
        "R2": "最大R2"
    })
    best = best.sort_values(["阶段", "作物", "省份"]).reset_index(drop=True)
    return best


# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("=" * 90)
    print("各省两阶段干旱指数 vs 冬春小麦气象单产（二次拟合）R2 计算")
    print("=" * 90)

    # ---------- 读取气象单产 ----------
    yield_long_df = load_yield_long(yield_file)
    print(f"\n✅ 气象单产长表读取完成，记录数: {len(yield_long_df)}")
    print(yield_long_df.head())

    # ---------- 阶段1 ----------
    result_stage1, merged_stage1 = compute_stage_r2(stage1_file, yield_long_df, "阶段1")

    # ---------- 阶段2 ----------
    result_stage2, merged_stage2 = compute_stage_r2(stage2_file, yield_long_df, "阶段2")

    # ---------- 合并 ----------
    result_all = pd.concat([result_stage1, result_stage2], ignore_index=True)
    result_all = result_all.sort_values(["阶段", "作物", "省份", "指数", "尺度"]).reset_index(drop=True)

    best_df = build_best_indicator_table(result_all)

    # =====================================================
    # 5. 保存结果
    # =====================================================
    out_stage1 = os.path.join(output_dir, "阶段1_R2结果_气象单产.xlsx")
    out_stage2 = os.path.join(output_dir, "阶段2_R2结果_气象单产.xlsx")
    out_all = os.path.join(output_dir, "两阶段R2汇总结果_气象单产.xlsx")

    with pd.ExcelWriter(out_stage1, engine="openpyxl") as writer:
        for crop in ["Spring_Wheat", "Winter_Wheat"]:
            sub = result_stage1[result_stage1["作物"] == crop].copy()
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)

        best1 = best_df[best_df["阶段"] == "阶段1"].copy()
        best1.to_excel(writer, sheet_name="阶段1最优指标", index=False)

    with pd.ExcelWriter(out_stage2, engine="openpyxl") as writer:
        for crop in ["Spring_Wheat", "Winter_Wheat"]:
            sub = result_stage2[result_stage2["作物"] == crop].copy()
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)

        best2 = best_df[best_df["阶段"] == "阶段2"].copy()
        best2.to_excel(writer, sheet_name="阶段2最优指标", index=False)

    with pd.ExcelWriter(out_all, engine="openpyxl") as writer:
        result_all.to_excel(writer, sheet_name="All_R2_Results", index=False)
        best_df.to_excel(writer, sheet_name="Best_Indicator_By_Province", index=False)

        for crop in ["Spring_Wheat", "Winter_Wheat"]:
            sub = result_all[result_all["作物"] == crop].copy()
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=f"{crop}_R2"[:31], index=False)

    print("\n" + "=" * 90)
    print("🎉 全部完成！")
    print("=" * 90)
    print(f"阶段1结果: {out_stage1}")
    print(f"阶段2结果: {out_stage2}")
    print(f"总结果  : {out_all}")


if __name__ == "__main__":
    main()