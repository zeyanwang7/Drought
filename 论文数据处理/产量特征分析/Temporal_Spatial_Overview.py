import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


# --- 1. 字体配置 ---
def apply_fonts():
    font_list = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'SimSun', 'Arial Unicode MS']
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            continue


apply_fonts()
sns.set_theme(style="ticks", rc={'font.sans-serif': plt.rcParams['font.sans-serif']})

# --- 2. 路径配置 ---
syrs_path = r'E:\data\计算结果\表格\单产标准化残差_SYRS.xlsx'
# 修改点：指向播种面积数据
area_path = r'E:\data\yeild_data\各省作物播种面积汇总_2002_2022.xlsx'
save_path_img = r'E:\data\计算结果\表格\SYRS_Area_Weighted_Time_Series.png'
save_path_csv = r'E:\data\计算结果\表格\全国加权平均SYRS数值.csv'

crops = ['小麦', '玉米', '水稻']
colors = {'小麦': '#1f77b4', '玉米': '#ff7f0e', '水稻': '#2ca02c'}


def get_cleaned_df(path, sheet):
    # 读取 Excel 并清洗索引和列名
    df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    # 处理列名：提取数字年份
    df.columns = df.columns.astype(str).str.extract('(\d+)')[0].astype(int)
    # 处理省份：只取前两个字进行模糊匹配，增强稳健性
    df.index = df.index.astype(str).str[:2]
    return df[~df.index.duplicated(keep='first')]


# --- 3. 计算与绘图 ---
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
results_to_save = pd.DataFrame()

for crop in crops:
    try:
        # 获取 SYRS 数据和对应的播种面积数据
        df_syrs = get_cleaned_df(syrs_path, crop)
        df_area = get_cleaned_df(area_path, crop)

        # 找年份和省份的交集（确保 2002-2022 对齐）
        common_years = df_syrs.columns.intersection(df_area.columns).sort_values()
        common_provs = df_syrs.index.intersection(df_area.index)

        s = df_syrs.loc[common_provs, common_years]
        a = df_area.loc[common_provs, common_years]

        # --- 核心修改：基于播种面积的动态加权平均 ---
        # 每一年的权重 = 该省面积 / 当年全国总面积
        weighted_mean = (s * a).sum(axis=0) / a.sum(axis=0)

        # 存储结果
        results_to_save[crop] = weighted_mean

        # 绘图
        ax.plot(weighted_mean.index, weighted_mean.values, label=f'{crop}',
                color=colors[crop], lw=2, marker='o', markersize=5)

        # 标注最低点（典型灾年）
        min_year = weighted_mean.idxmin()
        min_val = weighted_mean.min()
        ax.annotate(f'{min_year}', xy=(min_year, min_val), xytext=(0, -15),
                    textcoords='offset points', ha='center', fontsize=9, color=colors[crop])

    except Exception as e:
        print(f"处理 {crop} 失败: {e}")

# --- 4. 细节美化 ---
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.axhline(0, color='black', lw=1, ls='-')  # 0线
ax.axhline(-1, color='red', lw=0.8, ls='--', alpha=0.5)  # 异常减产阈值线

ax.set_title('全国主要作物播种面积加权平均单产标准化残差 (SYRS) 演变 (2002-2022)', fontsize=15, pad=15)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('面积加权平均 SYRS', fontsize=12)
ax.legend(loc='lower left', frameon=True)
ax.grid(True, axis='y', ls=':', alpha=0.5)

plt.tight_layout()
plt.savefig(save_path_img)
results_to_save.to_csv(save_path_csv)

print(f"\n[完成] 采用播种面积加权计算。")
print(f"[完成] 图表已保存至: {save_path_img}")
print(f"[完成] 数据表已导出至: {save_path_csv}")
plt.show()