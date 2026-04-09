# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# =========================================================
# 1. 输入输出路径
# =========================================================
input_excel = r"E:\data\Province_Stage_Index_Mean\Maize_各省两阶段平均干旱指数_2002_2022.xlsx"
output_dir = r"E:\data\Province_Stage_Index_Mean\整理后"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 2. 工具函数
# =========================================================
def standardize_crop_name(x):
    """统一作物名称"""
    if pd.isna(x):
        return x
    x = str(x).strip()
    mapping = {
        "Spring_Maize": "Spring_Maize",
        "Summer_Maize": "Summer_Maize",
        "春玉米": "Spring_Maize",
        "夏玉米": "Summer_Maize",
    }
    return mapping.get(x, x)


def standardize_scale_name(x):
    """统一尺度名称"""
    if pd.isna(x):
        return x
    x = str(x).strip()
    mapping = {
        "1month": "1month",
        "3month": "3month",
        "6month": "6month",
        "9month": "9month",
        "12month": "12month",
        "1个月": "1month",
        "3个月": "3month",
        "6个月": "6month",
        "9个月": "9month",
        "12个月": "12month",
    }
    return mapping.get(x, x)


def get_index_value_columns(index_name, columns):
    """
    根据当前行的“指数”字段，找到对应的阶段1/阶段2均值列
    例如 index_name='SPI' -> ('阶段1平均SPI', '阶段2平均SPI')
    """
    index_name = str(index_name).strip()

    stage1_col = f"阶段1平均{index_name}"
    stage2_col = f"阶段2平均{index_name}"

    if stage1_col not in columns or stage2_col not in columns:
        return None, None

    return stage1_col, stage2_col


def read_maize_all_results_to_long(file_path):
    """
    读取第二个代码生成的玉米 Excel -> All_Results
    并整理成标准长表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    df = pd.read_excel(file_path, sheet_name="All_Results")

    required_cols = ["年份", "省份", "作物", "指数", "尺度"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"All_Results 缺少必要字段: {c}")

    if "阶段1有效像元数" not in df.columns or "阶段2有效像元数" not in df.columns:
        raise ValueError("All_Results 缺少 阶段1有效像元数 / 阶段2有效像元数字段")

    # 统一格式
    df["年份"] = df["年份"].astype(int)
    df["省份"] = df["省份"].astype(str).str.strip()
    df["作物"] = df["作物"].apply(standardize_crop_name)
    df["尺度"] = df["尺度"].apply(standardize_scale_name)
    df["指数"] = df["指数"].astype(str).str.strip()

    long_records = []

    for _, row in df.iterrows():
        index_name = row["指数"]
        stage1_col, stage2_col = get_index_value_columns(index_name, df.columns)

        if stage1_col is None or stage2_col is None:
            print(f"⚠️ 未找到 {index_name} 对应的阶段均值列，跳过该行")
            continue

        v1 = row[stage1_col]
        v2 = row[stage2_col]
        n1 = row["阶段1有效像元数"]
        n2 = row["阶段2有效像元数"]

        # 阶段1
        if pd.notna(v1):
            long_records.append({
                "年份": row["年份"],
                "省份": row["省份"],
                "作物": row["作物"],
                "尺度": row["尺度"],
                "指数": index_name,
                "阶段": "阶段1",
                "指数值": v1,
                "有效像元数": n1
            })

        # 阶段2
        if pd.notna(v2):
            long_records.append({
                "年份": row["年份"],
                "省份": row["省份"],
                "作物": row["作物"],
                "尺度": row["尺度"],
                "指数": index_name,
                "阶段": "阶段2",
                "指数值": v2,
                "有效像元数": n2
            })

    df_long = pd.DataFrame(long_records)

    if len(df_long) == 0:
        raise ValueError("整理后长表为空，请检查 All_Results 的列名和指数字段是否一致。")

    # 增加月份尺度，便于排序
    df_long["月份尺度"] = (
        df_long["尺度"].astype(str)
        .str.replace("month", "", regex=False)
        .astype(int)
    )

    df_long = df_long[[
        "年份", "省份", "作物", "尺度", "月份尺度",
        "指数", "阶段", "指数值", "有效像元数"
    ]]

    return df_long


def build_wide_table(df_long):
    """
    长表 -> 宽表
    每行：年份、省份、作物、阶段
    每列：指数_尺度，如 SPI_1month、SPEI_3month
    """
    df_long = df_long.copy()
    df_long["指标名"] = df_long["指数"] + "_" + df_long["尺度"]

    df_wide = df_long.pivot_table(
        index=["年份", "省份", "作物", "阶段"],
        columns="指标名",
        values="指数值",
        aggfunc="first"
    ).reset_index()

    df_wide.columns.name = None

    meta_cols = ["年份", "省份", "作物", "阶段"]
    value_cols = [c for c in df_wide.columns if c not in meta_cols]

    def sort_key(col_name):
        # 例如 SPI_1month
        try:
            parts = col_name.split("_")
            index_name = parts[0]
            scale_num = int(parts[1].replace("month", ""))
            return (index_name, scale_num)
        except Exception:
            return (col_name, 999)

    value_cols = sorted(value_cols, key=sort_key)

    return df_wide[meta_cols + value_cols]


def check_missing_indices(df_long):
    """打印每个作物-阶段下包含了哪些指数，方便检查"""
    print("\n" + "=" * 80)
    print("检查各作物/阶段包含的指数：")
    print("=" * 80)

    summary = (
        df_long.groupby(["作物", "阶段"])["指数"]
        .unique()
        .reset_index()
    )

    for _, row in summary.iterrows():
        idx_list = sorted(list(row["指数"]))
        print(f"{row['作物']} | {row['阶段']} -> {idx_list}")


# =========================================================
# 3. 主程序
# =========================================================
def main():
    print("=" * 80)
    print("开始整理：玉米两阶段干旱指数宽表（修正版，保留所有指数）")
    print("=" * 80)

    # 读取并整理成长表
    df_long = read_maize_all_results_to_long(input_excel)

    # 排序
    df_long = df_long.sort_values(
        by=["作物", "省份", "年份", "阶段", "指数", "月份尺度"]
    ).reset_index(drop=True)

    check_missing_indices(df_long)

    # 保存长表
    long_output = os.path.join(output_dir, "玉米两阶段干旱指数_长表_2002_2022.xlsx")
    df_long.to_excel(long_output, index=False)
    print(f"\n✅ 长表已保存: {long_output}")

    # 生成总宽表
    df_wide = build_wide_table(df_long)
    wide_output = os.path.join(output_dir, "玉米两阶段干旱指数_宽表_2002_2022.xlsx")
    df_wide.to_excel(wide_output, index=False)
    print(f"✅ 总宽表已保存: {wide_output}")

    # 阶段1宽表
    df_stage1 = df_wide[df_wide["阶段"] == "阶段1"].copy().reset_index(drop=True)
    stage1_output = os.path.join(output_dir, "玉米阶段1干旱指数_宽表_2002_2022.xlsx")

    with pd.ExcelWriter(stage1_output, engine="openpyxl") as writer:
        for crop in ["Spring_Maize", "Summer_Maize"]:
            sub = df_stage1[df_stage1["作物"] == crop].copy().reset_index(drop=True)
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)
                print(f"✅ 阶段1 - {crop} 已写入，记录数: {len(sub)}")

    print(f"✅ 阶段1宽表文件已保存: {stage1_output}")

    # 阶段2宽表
    df_stage2 = df_wide[df_wide["阶段"] == "阶段2"].copy().reset_index(drop=True)
    stage2_output = os.path.join(output_dir, "玉米阶段2干旱指数_宽表_2002_2022.xlsx")

    with pd.ExcelWriter(stage2_output, engine="openpyxl") as writer:
        for crop in ["Spring_Maize", "Summer_Maize"]:
            sub = df_stage2[df_stage2["作物"] == crop].copy().reset_index(drop=True)
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)
                print(f"✅ 阶段2 - {crop} 已写入，记录数: {len(sub)}")

    print(f"✅ 阶段2宽表文件已保存: {stage2_output}")

    print("\n" + "=" * 80)
    print("🎉 全部整理完成（SPI / SPET / SPEI / SVPD / SEDI）")
    print("=" * 80)


if __name__ == "__main__":
    main()