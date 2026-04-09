import pandas as pd

# ===============================
# 1. 路径
# ===============================
pixel_file = r"E:\数据集\各省春夏玉米像元统计表.xlsx"
yield_file = r"C:\Users\wangrui\Desktop\作物产量\整理后的作物产量2002-2022.xlsx"
output_file = r"C:\Users\wangrui\Desktop\玉米春夏分解产量.xlsx"


# ===============================
# 2. 读取数据
# ===============================
pixel_df = pd.read_excel(pixel_file)
yield_df = pd.read_excel(yield_file)

# 清理列名空格
pixel_df.columns = pixel_df.columns.str.strip()
yield_df.columns = yield_df.columns.str.strip()

print("\n像元表字段：", pixel_df.columns.tolist())
print("产量表字段：", yield_df.columns.tolist())


# ===============================
# 3. 👉 强制指定“第一列为省份”（关键）
# ===============================
pixel_df.rename(columns={pixel_df.columns[0]: '省份'}, inplace=True)
yield_df.rename(columns={yield_df.columns[0]: '省份'}, inplace=True)


# ===============================
# 4. 👉 自动识别像元列
# ===============================
spring_col = [c for c in pixel_df.columns if '春' in c][0]
summer_col = [c for c in pixel_df.columns if '夏' in c][0]

pixel_df.rename(columns={
    spring_col: '春像元',
    summer_col: '夏像元'
}, inplace=True)


# ===============================
# 5. 👉 自动识别产量列
# ===============================
yield_col = None
for c in yield_df.columns:
    if '玉米' in c and '产量' in c:
        yield_col = c
        break
    if c == '玉米':
        yield_col = c

if yield_col is None:
    raise Exception("❌ 没找到玉米产量列，请检查")

yield_df.rename(columns={yield_col: '玉米产量'}, inplace=True)


# ===============================
# 6. 前2字匹配（核心）
# ===============================
def short_name(x):
    return str(x).strip()[:2]

pixel_df['key'] = pixel_df['省份'].apply(short_name)
yield_df['key'] = yield_df['省份'].apply(short_name)


# ===============================
# 7. 计算比例
# ===============================
pixel_df['总像元'] = pixel_df['春像元'] + pixel_df['夏像元']

pixel_df['春比例'] = pixel_df['春像元'] / pixel_df['总像元']
pixel_df['夏比例'] = pixel_df['夏像元'] / pixel_df['总像元']


# ===============================
# 8. 合并
# ===============================
df = pd.merge(
    yield_df,
    pixel_df[['key', '春比例', '夏比例']],
    on='key',
    how='left'
)


# ===============================
# 9. 检查匹配情况
# ===============================
missing = df[df['春比例'].isna()]
if len(missing) > 0:
    print("\n⚠️ 未匹配成功省份：")
    print(missing['省份'].unique())
else:
    print("\n✅ 所有省份匹配成功")


# ===============================
# 10. 计算产量
# ===============================
df['春玉米产量'] = df['玉米产量'] * df['春比例']
df['夏玉米产量'] = df['玉米产量'] * df['夏比例']


# ===============================
# 11. 保存
# ===============================
df.to_excel(output_file, index=False)

print("\n🎉 完成！输出：", output_file)