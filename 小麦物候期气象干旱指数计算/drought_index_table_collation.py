# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# =========================================================
# 1. 输入文件
# =========================================================
input_files = {
    "SPEI": r"E:\data\Province_Stage_Index_Mean\各省SPEI两阶段平均值_2002_2022.xlsx",
    "SPI":  r"E:\data\Province_Stage_Index_Mean\各省SPI两阶段平均值_2002_2022.xlsx",
    "SVPD": r"E:\data\Province_Stage_Index_Mean\各省SVPD两阶段平均值_2002_2022.xlsx",
    "SEDI": r"E:\data\Province_Stage_Index_Mean\各省SEDI两阶段平均值_2002_2022.xlsx",
    "SPET": r"E:\data\Province_Stage_Index_Mean\各省SPET两阶段平均值_2002_2022.xlsx",
}

# 输出目录
output_dir = r"E:\data\Province_Stage_Index_Mean\整理后"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. 工具函数
# =========================================================
def standardize_crop_name(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    mapping = {
        "Spring_Wheat": "Spring_Wheat",
        "Winter_Wheat": "Winter_Wheat",
        "春小麦": "Spring_Wheat",
        "冬小麦": "Winter_Wheat",
    }
    return mapping.get(x, x)


def standardize_scale_name(x):
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


def read_one_index_excel(index_name, file_path):
    """读取一个指数 Excel 的 All_Results，并整理成长表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    df = pd.read_excel(file_path, sheet_name="All_Results")

    required_base = ["年份", "省份", "作物", "尺度"]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"{file_path} 缺少必要字段: {c}")

    stage1_col = f"阶段1平均{index_name}"
    stage2_col = f"阶段2平均{index_name}"
    pix1_col = "阶段1有效像元数"
    pix2_col = "阶段2有效像元数"

    for c in [stage1_col, stage2_col, pix1_col, pix2_col]:
        if c not in df.columns:
            raise ValueError(f"{file_path} 缺少字段: {c}")

    df["作物"] = df["作物"].apply(standardize_crop_name)
    df["尺度"] = df["尺度"].apply(standardize_scale_name)
    df["年份"] = df["年份"].astype(int)
    df["省份"] = df["省份"].astype(str).str.strip()

    # 阶段1
    df_stage1 = df[["年份", "省份", "作物", "尺度", stage1_col, pix1_col]].copy()
    df_stage1.columns = ["年份", "省份", "作物", "尺度", "指数值", "有效像元数"]
    df_stage1["阶段"] = "阶段1"
    df_stage1["指数"] = index_name

    # 阶段2
    df_stage2 = df[["年份", "省份", "作物", "尺度", stage2_col, pix2_col]].copy()
    df_stage2.columns = ["年份", "省份", "作物", "尺度", "指数值", "有效像元数"]
    df_stage2["阶段"] = "阶段2"
    df_stage2["指数"] = index_name

    df_long = pd.concat([df_stage1, df_stage2], ignore_index=True)
    df_long = df_long.dropna(subset=["指数值"]).reset_index(drop=True)

    # 增加数值型尺度，方便排序
    df_long["月份尺度"] = df_long["尺度"].str.replace("month", "", regex=False).astype(int)

    df_long = df_long[["年份", "省份", "作物", "尺度", "月份尺度", "指数", "阶段", "指数值", "有效像元数"]]

    return df_long


def build_wide_table(df_long):
    """将长表转为宽表"""
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

    def sort_key(x):
        parts = x.split("_")
        # 例如 SPEI_1month
        index_name = parts[0]
        scale_num = int(parts[1].replace("month", ""))
        return (index_name, scale_num)

    value_cols = sorted(value_cols, key=sort_key)
    df_wide = df_wide[meta_cols + value_cols]

    return df_wide


# =========================================================
# 3. 主程序
# =========================================================
def main():
    print("=" * 80)
    print("开始整理：小麦两阶段干旱指数宽表（含 SPET）")
    print("=" * 80)

    all_long_list = []

    for index_name, file_path in input_files.items():
        print(f"\n正在读取: {index_name}")
        df_long_one = read_one_index_excel(index_name, file_path)
        all_long_list.append(df_long_one)
        print(f"  -> {index_name} 读取完成，记录数: {len(df_long_one)}")

    # 合并总长表
    df_long_all = pd.concat(all_long_list, ignore_index=True)
    df_long_all = df_long_all.sort_values(
        by=["作物", "省份", "年份", "阶段", "指数", "月份尺度"]
    ).reset_index(drop=True)

    # 保存总长表
    long_output = os.path.join(output_dir, "小麦两阶段干旱指数_长表_2002_2022.xlsx")
    df_long_all.to_excel(long_output, index=False)
    print(f"\n✅ 长表已保存: {long_output}")

    # 总宽表
    df_wide_all = build_wide_table(df_long_all)
    wide_output = os.path.join(output_dir, "小麦两阶段干旱指数_宽表_2002_2022.xlsx")
    df_wide_all.to_excel(wide_output, index=False)
    print(f"✅ 总宽表已保存: {wide_output}")

    # =====================================================
    # 阶段1宽表：春小麦、冬小麦分别写入两个工作表
    # =====================================================
    df_stage1 = df_wide_all[df_wide_all["阶段"] == "阶段1"].copy().reset_index(drop=True)
    stage1_output = os.path.join(output_dir, "小麦阶段1干旱指数_宽表_2002_2022.xlsx")

    with pd.ExcelWriter(stage1_output, engine="openpyxl") as writer:
        for crop in ["Spring_Wheat", "Winter_Wheat"]:
            sub = df_stage1[df_stage1["作物"] == crop].copy().reset_index(drop=True)
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)
                print(f"✅ 阶段1 - {crop} 工作表已写入，记录数: {len(sub)}")

    print(f"✅ 阶段1宽表文件已保存: {stage1_output}")

    # =====================================================
    # 阶段2宽表：春小麦、冬小麦分别写入两个工作表
    # =====================================================
    df_stage2 = df_wide_all[df_wide_all["阶段"] == "阶段2"].copy().reset_index(drop=True)
    stage2_output = os.path.join(output_dir, "小麦阶段2干旱指数_宽表_2002_2022.xlsx")

    with pd.ExcelWriter(stage2_output, engine="openpyxl") as writer:
        for crop in ["Spring_Wheat", "Winter_Wheat"]:
            sub = df_stage2[df_stage2["作物"] == crop].copy().reset_index(drop=True)
            if len(sub) > 0:
                sub.to_excel(writer, sheet_name=crop, index=False)
                print(f"✅ 阶段2 - {crop} 工作表已写入，记录数: {len(sub)}")

    print(f"✅ 阶段2宽表文件已保存: {stage2_output}")

    print("\n" + "=" * 80)
    print("🎉 全部整理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()