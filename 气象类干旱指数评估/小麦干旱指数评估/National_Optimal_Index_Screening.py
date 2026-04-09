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
# 你前面计算好的 R2 总结果表
r2_file = r"E:\data\Province_Stage_Index_Mean\R2结果_气象单产\两阶段R2汇总结果_气象单产.xlsx"

# 多年产量表
yield_file = r"C:\Users\wangrui\Desktop\作物产量\整理后的作物产量2002-2022.xlsx"

# 输出目录
output_dir = r"E:\data\Province_Stage_Index_Mean\最终筛选结果"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "冬春小麦_全国加权平均R2最优指标_按产量加权.xlsx")


# =========================================================
# 2. 基础参数
# =========================================================
YEARS = list(range(2002, 2023))

# 产量表工作表名称映射
yield_sheet_map = {
    "Winter_Wheat": "冬小麦产量(万吨)",
    "Spring_Wheat": "春小麦产量(万吨)"
}

# 作物中英文映射
crop_cn_map = {
    "Winter_Wheat": "冬小麦",
    "Spring_Wheat": "春小麦"
}


# =========================================================
# 3. 工具函数
# =========================================================
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")


def normalize_province_name(name):
    """
    省份名称标准化
    """
    if pd.isna(name):
        return np.nan

    name = str(name).strip()
    name = re.sub(r"\s+", "", name)

    mapping = {
        "北京": "北京市", "北京市": "北京市",
        "天津": "天津市", "天津市": "天津市",
        "上海": "上海市", "上海市": "上海市",
        "重庆": "重庆市", "重庆市": "重庆市",

        "河北": "河北省", "河北省": "河北省",
        "山西": "山西省", "山西省": "山西省",
        "辽宁": "辽宁省", "辽宁省": "辽宁省",
        "吉林": "吉林省", "吉林省": "吉林省",
        "黑龙江": "黑龙江省", "黑龙江省": "黑龙江省",
        "江苏": "江苏省", "江苏省": "江苏省",
        "浙江": "浙江省", "浙江省": "浙江省",
        "安徽": "安徽省", "安徽省": "安徽省",
        "福建": "福建省", "福建省": "福建省",
        "江西": "江西省", "江西省": "江西省",
        "山东": "山东省", "山东省": "山东省",
        "河南": "河南省", "河南省": "河南省",
        "湖北": "湖北省", "湖北省": "湖北省",
        "湖南": "湖南省", "湖南省": "湖南省",
        "广东": "广东省", "广东省": "广东省",
        "海南": "海南省", "海南省": "海南省",
        "四川": "四川省", "四川省": "四川省",
        "贵州": "贵州省", "贵州省": "贵州省",
        "云南": "云南省", "云南省": "云南省",
        "陕西": "陕西省", "陕西省": "陕西省",
        "甘肃": "甘肃省", "甘肃省": "甘肃省",
        "青海": "青海省", "青海省": "青海省",

        "内蒙古": "内蒙古自治区", "内蒙古自治区": "内蒙古自治区",
        "广西": "广西壮族自治区", "广西壮族自治区": "广西壮族自治区",
        "西藏": "西藏自治区", "西藏自治区": "西藏自治区",
        "宁夏": "宁夏回族自治区", "宁夏回族自治区": "宁夏回族自治区",
        "新疆": "新疆维吾尔自治区", "新疆维吾尔自治区": "新疆维吾尔自治区",

        "内蒙": "内蒙古自治区",
    }

    if name in mapping:
        return mapping[name]

    short = name[:2]
    for k, v in mapping.items():
        if k[:2] == short or v[:2] == short:
            return v

    return name


def detect_year_column(df):
    """
    自动识别年份列
    """
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in ["年份", "year", "Year", "YEAR", "Unnamed: 0"]:
            return col

    for col in df.columns:
        vals = df[col].dropna().head(10).tolist()
        ok_count = 0
        for v in vals:
            try:
                iv = int(v)
                if 1900 <= iv <= 2100:
                    ok_count += 1
            except:
                pass
        if ok_count >= max(3, len(vals) // 2):
            return col

    raise ValueError("未能识别产量表中的年份列")


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
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df["作物中文"] = df["作物"].map(crop_cn_map)

    df = df.dropna(subset=["R2", "省份", "作物中文"]).reset_index(drop=True)

    return df


# =========================================================
# 5. 读取产量表并计算权重
# =========================================================
def build_yield_weight_table(yield_excel, crop_code):
    """
    从产量表中读取某个作物的省级产量，
    计算 2002-2022 多年平均产量和产量权重
    """
    if crop_code not in yield_sheet_map:
        raise ValueError(f"不支持的作物代码: {crop_code}")

    sheet_name = yield_sheet_map[crop_code]
    df = pd.read_excel(yield_excel, sheet_name=sheet_name)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    year_col = detect_year_column(df)
    df = df.rename(columns={year_col: "年份"})

    df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
    df = df[df["年份"].isin(YEARS)].copy()

    province_cols = [c for c in df.columns if c != "年份"]

    df_long = df.melt(
        id_vars="年份",
        value_vars=province_cols,
        var_name="省份",
        value_name="产量"
    )

    df_long["省份"] = df_long["省份"].apply(normalize_province_name)
    df_long["产量"] = pd.to_numeric(df_long["产量"], errors="coerce")
    df_long = df_long.dropna(subset=["省份", "产量"]).copy()

    avg_yield = (
        df_long.groupby("省份", as_index=False)["产量"]
        .mean()
        .rename(columns={"产量": "多年平均产量_万吨"})
    )

    total = avg_yield["多年平均产量_万吨"].sum()
    if total == 0 or pd.isna(total):
        raise ValueError(f"{crop_code} 的多年平均产量总和为0，无法计算权重")

    avg_yield["产量权重"] = avg_yield["多年平均产量_万吨"] / total
    avg_yield["作物"] = crop_code
    avg_yield["作物中文"] = crop_cn_map[crop_code]

    avg_yield = avg_yield.sort_values("产量权重", ascending=False).reset_index(drop=True)

    return avg_yield, df_long


# =========================================================
# 6. 合并 R2 与产量权重
# =========================================================
def merge_r2_with_weight(df_r2, yield_excel):
    all_merge = []
    all_weight = []

    for crop_code in ["Winter_Wheat", "Spring_Wheat"]:
        sub_r2 = df_r2[df_r2["作物"] == crop_code].copy()
        if len(sub_r2) == 0:
            continue

        weight_df, yield_long_df = build_yield_weight_table(yield_excel, crop_code)

        merged = pd.merge(
            sub_r2,
            weight_df[["省份", "多年平均产量_万吨", "产量权重"]],
            on="省份",
            how="left"
        )

        unmatched = merged[merged["产量权重"].isna()]["省份"].drop_duplicates().tolist()
        if unmatched:
            print(f"\n⚠ {crop_cn_map[crop_code]} 以下省份未匹配到产量权重：")
            for p in unmatched:
                print("   -", p)

        merged = merged.dropna(subset=["R2", "产量权重"]).copy()

        all_merge.append(merged)
        all_weight.append(weight_df)

    merge_all = pd.concat(all_merge, ignore_index=True) if all_merge else pd.DataFrame()
    weight_all = pd.concat(all_weight, ignore_index=True) if all_weight else pd.DataFrame()

    return merge_all, weight_all


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
        merged_df.groupby(["作物中文", "作物", "阶段", "指数", "尺度"], as_index=False)
        .apply(lambda g: pd.Series({
            "参与省份数": g["省份"].nunique(),
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
        .groupby(["作物中文", "作物", "阶段"], as_index=False)
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
    print("开始生成：冬春小麦 各指标全国尺度产量加权平均R2 + 全国最优指标")
    print("=" * 100)

    # 1) 读取各省各指标R2结果
    df_r2 = load_r2_result(r2_file)
    print(f"✅ R2结果读取完成，记录数: {len(df_r2)}")

    # 2) 与各省产量权重合并
    df_merge, df_weight = merge_r2_with_weight(df_r2, yield_file)
    print("✅ 各省指标R2与产量权重合并完成")

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
            df_weight.to_excel(writer, sheet_name="各省多年平均产量权重", index=False)

        if len(df_merge) > 0:
            df_merge.to_excel(writer, sheet_name="R2值_权重合并明细", index=False)

        if len(df_weighted_r2) > 0:
            df_weighted_r2.to_excel(writer, sheet_name="各指标全国加权平均R2排名", index=False)

        if len(df_best) > 0:
            df_best.to_excel(writer, sheet_name="全国最优指标汇总", index=False)

        # 分作物分阶段输出
        for crop in ["Winter_Wheat", "Spring_Wheat"]:
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