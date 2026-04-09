import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 中文字体设置（适配 Windows / PyCharm）
# =============================================================================
def setup_chinese_font():
    """
    自动设置 matplotlib 中文字体，适配 Windows 环境。
    按顺序尝试多个常见中文字体。
    """
    candidate_fonts = [
        "Microsoft YaHei",   # 微软雅黑
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "KaiTi",             # 楷体
        "FangSong",          # 仿宋
        "NSimSun",
        "Arial Unicode MS"
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    selected_font = None
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is not None:
        mpl.rcParams["font.sans-serif"] = [selected_font]
        mpl.rcParams["font.family"] = "sans-serif"
        print(f"✅ 已设置中文字体: {selected_font}")
    else:
        print("⚠️ 未找到常见中文字体，中文可能显示异常。")
        print("建议在 Windows 中安装：Microsoft YaHei 或 SimHei")

    # 解决负号显示为方块的问题
    mpl.rcParams["axes.unicode_minus"] = False

    # 可选：统一字号风格
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10

setup_chinese_font()

# =============================================================================
# 1. 路径设置
# =============================================================================
input_excel = r"E:\data\Province_Stage_Index_Mean\最终筛选结果\冬春小麦_全国加权平均R2最优指标_按产量加权.xlsx"
input_sheet = "R2值_权重合并明细"

output_dir = r"E:\data\Province_Stage_Index_Mean\最终筛选结果\稳定性_解释力散点图"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(output_dir, "冬春小麦_稳定性_解释力散点图.png")
output_excel = os.path.join(output_dir, "冬春小麦_稳定性_解释力散点图_统计结果.xlsx")

# =============================================================================
# 2. 参数设置
# =============================================================================
INDEX_ORDER = ["SPI", "SPEI", "SPET", "SVPD", "SEDI"]
CROP_ORDER = ["Winter_Wheat", "Spring_Wheat"]
STAGE_ORDER = ["阶段1", "阶段2"]

CROP_NAME_MAP = {
    "Winter_Wheat": "冬小麦",
    "Spring_Wheat": "春小麦"
}

INDEX_COLOR_MAP = {
    "SPI":  "#d73027",
    "SPEI": "#fc8d59",
    "SPET": "#91bfdb",
    "SVPD": "#4575b4",
    "SEDI": "#66bd63",
}

MIN_SIZE = 60
SIZE_SCALE = 2200

# =============================================================================
# 3. 读取数据
# =============================================================================
print("=" * 100)
print("开始读取 Excel 数据")
print("=" * 100)

df = pd.read_excel(input_excel, sheet_name=input_sheet)

print(f"读取完成，共 {len(df)} 条记录")
print("字段如下：")
print(df.columns.tolist())

# =============================================================================
# 4. 数据检查与预处理
# =============================================================================
required_cols = ["阶段", "作物", "省份", "指数", "尺度", "R2", "产量权重"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要字段: {missing_cols}")

df = df.dropna(subset=required_cols).copy()

df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
df["产量权重"] = pd.to_numeric(df["产量权重"], errors="coerce")
df["尺度"] = pd.to_numeric(df["尺度"], errors="coerce")

df = df.dropna(subset=["R2", "产量权重", "尺度"]).copy()

print(f"\n清洗后剩余 {len(df)} 条记录")

# =============================================================================
# 5. 计算全国加权平均 R2
# =============================================================================
print("\n" + "=" * 100)
print("开始计算全国加权平均 R²")
print("=" * 100)

weighted_df = (
    df.groupby(["作物", "阶段", "指数", "尺度"], as_index=False)
      .apply(lambda g: pd.Series({
          "全国加权R2": np.average(g["R2"], weights=g["产量权重"])
      }))
      .reset_index(drop=True)
)

print(weighted_df.head())

# =============================================================================
# 6. 计算省际稳定性
# =============================================================================
print("\n" + "=" * 100)
print("开始计算省际稳定性指标")
print("=" * 100)

stability_df = (
    df.groupby(["作物", "阶段", "指数", "尺度"], as_index=False)
      .agg(
          省际R2均值=("R2", "mean"),
          省际R2标准差=("R2", "std"),
          省际R2中位数=("R2", "median"),
          Q1=("R2", lambda x: np.percentile(x, 25)),
          Q3=("R2", lambda x: np.percentile(x, 75)),
          省份数=("省份", "nunique")
      )
)

stability_df["省际R2标准差"] = stability_df["省际R2标准差"].fillna(0)
stability_df["IQR"] = stability_df["Q3"] - stability_df["Q1"]

print(stability_df.head())

# =============================================================================
# 7. 提取每个省的最优组合（R2最大）
# =============================================================================
print("\n" + "=" * 100)
print("开始提取各省最优组合")
print("=" * 100)

group_cols = ["作物", "阶段", "省份"]
idx = df.groupby(group_cols)["R2"].idxmax()
best_df = df.loc[idx].copy()

best_df = best_df.sort_values(["作物", "阶段", "省份"]).reset_index(drop=True)

print(f"共提取 {len(best_df)} 条省级最优记录")
print(best_df.head())

win_df = (
    best_df.groupby(["作物", "阶段", "指数", "尺度"], as_index=False)
           .agg(
               最优覆盖省份数=("省份", "nunique"),
               最优覆盖产量权重=("产量权重", "sum")
           )
)

print(win_df.head())

# =============================================================================
# 8. 合并绘图数据
# =============================================================================
print("\n" + "=" * 100)
print("开始合并绘图统计表")
print("=" * 100)

plot_df = weighted_df.merge(
    stability_df,
    on=["作物", "阶段", "指数", "尺度"],
    how="left"
)

plot_df = plot_df.merge(
    win_df,
    on=["作物", "阶段", "指数", "尺度"],
    how="left"
)

plot_df["最优覆盖省份数"] = plot_df["最优覆盖省份数"].fillna(0)
plot_df["最优覆盖产量权重"] = plot_df["最优覆盖产量权重"].fillna(0)
plot_df["气泡大小"] = MIN_SIZE + plot_df["最优覆盖产量权重"] * SIZE_SCALE

print(plot_df.head(10))

# =============================================================================
# 9. 导出统计结果
# =============================================================================
print("\n" + "=" * 100)
print("导出统计结果")
print("=" * 100)

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="原始明细", index=False)
    weighted_df.to_excel(writer, sheet_name="全国加权R2", index=False)
    stability_df.to_excel(writer, sheet_name="省际稳定性", index=False)
    best_df.to_excel(writer, sheet_name="各省最优组合", index=False)
    win_df.to_excel(writer, sheet_name="最优覆盖统计", index=False)
    plot_df.to_excel(writer, sheet_name="散点图绘图表", index=False)

print(f"统计结果已保存：{output_excel}")

# =============================================================================
# 10. 绘图函数（保存 + PyCharm显示）
# =============================================================================
def plot_stability_explanatory_scatter(plot_df, output_png):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_order = [
        ("Winter_Wheat", "阶段1"),
        ("Winter_Wheat", "阶段2"),
        ("Spring_Wheat", "阶段1"),
        ("Spring_Wheat", "阶段2"),
    ]

    # =========================
    # 1. 主图散点
    # =========================
    for ax, (crop, stage) in zip(axes, plot_order):
        sub = plot_df[
            (plot_df["作物"] == crop) &
            (plot_df["阶段"] == stage)
        ].copy()

        sub["指数"] = pd.Categorical(sub["指数"], categories=INDEX_ORDER, ordered=True)
        sub = sub.sort_values(["指数", "尺度"])

        for idx_name in INDEX_ORDER:
            tmp = sub[sub["指数"] == idx_name].copy()
            if tmp.empty:
                continue

            ax.scatter(
                tmp["全国加权R2"],
                tmp["省际R2标准差"],
                s=tmp["气泡大小"],
                c=INDEX_COLOR_MAP[idx_name],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
                zorder=2
            )

            # 尺度数字：放在气泡中心，取消白底/描边
            for _, row in tmp.iterrows():
                ax.text(
                    row["全国加权R2"],
                    row["省际R2标准差"],
                    f"{int(row['尺度'])}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    zorder=3
                )

        ax.set_title(f"{CROP_NAME_MAP.get(crop, crop)} - {stage}")
        ax.set_xlabel("全国加权平均 R²")
        ax.set_ylabel("省际 R² 标准差")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=1)

        if len(sub) > 0:
            x_min = max(0, sub["全国加权R2"].min() - 0.01)
            x_max = sub["全国加权R2"].max() + 0.02
            y_min = max(0, sub["省际R2标准差"].min() - 0.01)
            y_max = sub["省际R2标准差"].max() + 0.02
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # =========================
    # 2. 下方颜色图例（统一大小）
    # =========================
    color_handles = []
    for idx_name in INDEX_ORDER:
        h = plt.scatter(
            [], [],
            s=120,
            color=INDEX_COLOR_MAP[idx_name],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85
        )
        color_handles.append(h)

    legend_color = fig.legend(
        handles=color_handles,
        labels=INDEX_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.25, 0.03),   # 左半部分
        ncol=5,
        frameon=False,
        fontsize=11,
        title="干旱指数",
        title_fontsize=12,
        handletextpad=0.6,
        columnspacing=1.4
    )

    # =========================
    # 3. 下方气泡大小图例（同行排列）
    # =========================
    qs = plot_df["最优覆盖产量权重"].quantile([0.25, 0.50, 0.75]).round(2).tolist()
    weight_levels = sorted(list(set([w for w in qs if w > 0])))

    if len(weight_levels) < 3:
        weight_levels = [0.05, 0.10, 0.20,0.30,0.40]

    size_handles = []
    size_labels = []
    for w in weight_levels:
        s = MIN_SIZE + w * SIZE_SCALE
        h = plt.scatter(
            [], [],
            s=s,
            color="lightgray",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8
        )
        size_handles.append(h)
        size_labels.append(f"{w:.2f}")

    legend_size = fig.legend(
        handles=size_handles,
        labels=size_labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.03),   # 右半部分
        ncol=len(size_labels),
        frameon=False,
        fontsize=11,
        title="最优覆盖产量权重",
        title_fontsize=12,
        handletextpad=1.0,
        columnspacing=1.8
    )

    # =========================
    # 4. 总标题与布局
    # =========================
    plt.suptitle("全国代表性—省际异质性权衡散点图", fontsize=17, y=0.98)

    # 下方给双图例留空间
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"散点图已保存：{output_png}")

    plt.show()
    plt.close()

# =============================================================================
# 11. 开始绘图
# =============================================================================
print("\n" + "=" * 100)
print("开始绘制稳定性—解释力散点图")
print("=" * 100)

plot_stability_explanatory_scatter(plot_df, output_png)

# =============================================================================
# 12. 结束
# =============================================================================
print("\n" + "=" * 100)
print("全部完成")
print("=" * 100)