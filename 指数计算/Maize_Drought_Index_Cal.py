import subprocess
import sys
import os
import time
from datetime import datetime

# =========================================
# 1. 这里改成你自己的 Python 解释器路径
# =========================================
python_exe = r"D:\anaconda\envs\corn_drought\python.exe"

# =========================================
# 2. 这里改成你5个脚本的真实路径
#    按你希望的执行顺序排列
# =========================================
scripts = [
    ("SPI计算",  r"D:\pyproject\指数计算\SPI\SPI_Maize_Auto.py"),
    ("SPEI计算", r"D:\pyproject\指数计算\SPEI\SPEI_Maize_Auto.py"),
    ("SEDI计算", r"D:\pyproject\指数计算\SEDI\SEDI_Maize_Auto.py"),
    ("SVPD计算", r"D:\pyproject\指数计算\SVPD\SVPD_Maize_Auto.py"),
    ("SPET计算", r"D:\pyproject\指数计算\SPET\SPET_Maize_Auto.py"),
]

# =========================================
# 3. 日志文件夹
# =========================================
log_dir = r"D:\pyproject\运行日志"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"maize_drought_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def write_log(msg):
    """同时输出到屏幕和日志文件"""
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_script(script_name, script_path):
    """运行单个脚本，实时输出日志"""
    write_log("\n" + "=" * 80)
    write_log(f"🚀 开始执行: {script_name}")
    write_log(f"📄 脚本路径: {script_path}")
    write_log(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("=" * 80)

    if not os.path.exists(script_path):
        write_log(f"❌ 脚本不存在: {script_path}")
        return False

    start_time = time.time()

    try:
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace"   # 关键：避免编码报错中断
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                write_log(line)

        process.wait()

        end_time = time.time()
        elapsed = end_time - start_time

        if process.returncode == 0:
            write_log("-" * 80)
            write_log(f"✅ {script_name} 执行完成")
            write_log(f"⏱ 用时: {elapsed/60:.2f} 分钟")
            write_log(f"🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            write_log("-" * 80)
            return True
        else:
            write_log("-" * 80)
            write_log(f"❌ {script_name} 执行失败")
            write_log(f"返回码: {process.returncode}")
            write_log(f"⏱ 已运行: {elapsed/60:.2f} 分钟")
            write_log("-" * 80)
            return False

    except Exception as e:
        write_log(f"❌ 运行 {script_name} 时发生异常: {e}")
        return False


def main():
    total_start = time.time()

    write_log("=" * 100)
    write_log("🌽 春夏玉米干旱指数自动计算任务启动")
    write_log(f"🕒 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(f"🐍 Python解释器: {python_exe}")
    write_log(f"📝 日志文件: {log_file}")
    write_log("=" * 100)

    for script_name, script_path in scripts:
        ok = run_script(script_name, script_path)
        if not ok:
            write_log("\n⛔ 任务终止：前一脚本执行失败，后续脚本不再继续运行。")
            sys.exit(1)

    total_end = time.time()
    total_elapsed = total_end - total_start

    write_log("\n" + "=" * 100)
    write_log("🎉 所有脚本已按顺序执行完成！")
    write_log(f"⏱ 总耗时: {total_elapsed/3600:.2f} 小时")
    write_log(f"🕒 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("=" * 100)


if __name__ == "__main__":
    main()