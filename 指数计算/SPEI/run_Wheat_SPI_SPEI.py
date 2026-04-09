import subprocess
import time

# ===============================
# 1. 文件路径（改成你自己的）
# ===============================
spi_script = r"D:\pyproject\指数计算\SPI\SPI_Wheat_Auto.py "
spei_script = r"D:\pyproject\指数计算\SPEI\SPEI_Wheat_Auto.py "

# ===============================
# 2. 运行函数
# ===============================
def run_script(script_path, name):
    print("\n" + "=" * 60)
    print(f"🚀 开始执行: {name}")
    print("=" * 60)

    start = time.time()

    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True
    )

    end = time.time()

    if result.returncode != 0:
        print(f"❌ {name} 执行失败！")
        exit(1)

    print(f"✅ {name} 执行完成，用时 {(end - start)/60:.2f} 分钟")


# ===============================
# 3. 主流程
# ===============================
if __name__ == "__main__":

    total_start = time.time()

    print("\n🌾 开始全流程计算（SPI → SPEI）")

    # Step1: SPI
    run_script(spi_script, "SPI计算")

    # Step2: SPEI
    run_script(spei_script, "SPEI计算")

    total_end = time.time()

    print("\n" + "=" * 60)
    print("🎉 全部任务完成！")
    print(f"⏱ 总耗时: {(total_end - total_start)/3600:.2f} 小时")
    print("=" * 60)