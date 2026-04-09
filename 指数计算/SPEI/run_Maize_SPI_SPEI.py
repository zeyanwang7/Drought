import subprocess

def run_realtime(cmd, name):
    print(f"\n🚀 正在运行: {name}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",  # ✅ 强制UTF-8
        errors="ignore"  # ✅ 防止异常字符崩溃
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"{name} 运行失败")

    print(f"\n✅ {name} 完成\n")


if __name__ == "__main__":

    python_exe = r"D:\anaconda\envs\corn_drought\python.exe"

    spi_script = r"D:\pyproject\指数计算\SPI\SPI_Maize_Auto.py"
    spei_script = r"D:\pyproject\指数计算\SPEI\SPEI_Maize_Auto.py"

    run_realtime([python_exe, spi_script], "SPI计算")
    run_realtime([python_exe, spei_script], "SPEI计算")

    print("🎉 全部完成")