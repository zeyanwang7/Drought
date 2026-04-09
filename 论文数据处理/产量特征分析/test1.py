import pandas as pd
import os

# --- 1. 路径配置 ---
folder_path = r'E:\数据集\作物播种面积'
save_path = r'E:\数据集\各省作物播种面积汇总_2002_2022.xlsx'

# 作物配置：键为 Sheet 名，值为 Excel 索引 (行索引, 预期包含的名称)
# 注意：Excel 第5行对应 index 4，第6行对应 5，第7行对应 6
check_config = {
    '水稻': (4, '稻谷'),
    '小麦': (5, '小麦'),
    '玉米': (6, '玉米')
}

years_desc = [str(y) for y in range(2022, 2001, -1)]
all_data = {'水稻': [], '小麦': [], '玉米': []}
invalid_provinces = []  # 用于存放格式不正确的省份

# --- 2. 遍历并提取数据 ---
print("开始提取数据并进行监测...")

files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

for file_name in files:
    file_full_path = os.path.join(folder_path, file_name)
    province = os.path.splitext(file_name)[0]

    try:
        # 读取文件，不设表头
        df = pd.read_excel(file_full_path, header=None)

        # --- 监测机制 ---
        is_valid = True
        for crop, (row_idx, expected_text) in check_config.items():
            # 获取 A 列该行的内容 (iloc[row_idx, 0])
            actual_text = str(df.iloc[row_idx, 0])
            if expected_text not in actual_text:
                print(f"  [警告] {province} 格式异常: 第{row_idx + 1}行应包含'{expected_text}'，实际为'{actual_text}'")
                is_valid = False

        if not is_valid:
            invalid_provinces.append(province)
            # 虽然格式不对，但你可以选择继续提取（如果只是表头写错了），或者跳过。
            # 这里选择继续提取，但会在终端显著提醒。

        # --- 数据提取 ---
        for crop, (row_idx, _) in check_config.items():
            # B列(1) 到 V列(21)
            row_values = df.iloc[row_idx, 1:22].values

            # 转为 Series 并按年份升序 (2002 -> 2022)
            s = pd.Series(row_values, index=years_desc).sort_index()

            # 整合数据
            item = {'省份': province}
            item.update(s.to_dict())
            all_data[crop].append(item)

    except Exception as e:
        print(f"  [错误] 无法读取文件 {file_name}: {e}")

# --- 3. 汇总与结果反馈 ---
print("\n--- 处理报告 ---")
if invalid_provinces:
    print(f"以下省份的 Excel 格式不符合监测要求 (已打印具体行):")
    print(", ".join(invalid_provinces))
else:
    print("所有省份文件校验通过！")

with pd.ExcelWriter(save_path) as writer:
    for crop in all_data:
        final_df = pd.DataFrame(all_data[crop])
        # 排序年份列
        cols = ['省份'] + [str(y) for y in range(2002, 2023)]
        final_df = final_df[cols]
        final_df.to_excel(writer, sheet_name=crop, index=False)

print(f"\n[任务完成] 汇总文件已生成: {save_path}")