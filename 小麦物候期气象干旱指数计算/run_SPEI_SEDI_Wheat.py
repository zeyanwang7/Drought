import subprocess
import time
import sys
import os

# ===============================
# 1. 文件路径（改成你自己的）
# ===============================
spei_script = r"D:\pyproject\物候期干旱指数计算\Wheat_Phenological_SPEI.py"
sedi_script = r"D:\pyproject\物候期干旱指数计算\Wheat_Phenological_SEDI.py"

# ===============================
# 2. 运行函数
# ===============================
def run_script(script_path, name):
    print("\n" + "=" * 60)
    print(f"🚀 开始执行: {name}")
    print(f"📄 脚本路径: {script_path}")
    print("=" * 60)

    if not os.path.exists(script_path):
        print(f"❌ 脚本不存在: {script_path}")
        sys.exit(1)

    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )

    end = time.time()

    if result.returncode != 0:
        print(f"❌ {name} 执行失败！")
        sys.exit(1)

    print(f"✅ {name} 执行完成，用时 {(end - start)/60:.2f} 分钟")


# ===============================
# 3. 主流程
# ===============================
if __name__ == "__main__":

    total_start = time.time()

    print("\n🌾 开始全流程计算（SPEI → SEDI）")

    # Step1: SPEI
    run_script(spei_script, "SPEI计算")

    # Step2: SEDI
    run_script(sedi_script, "SEDI计算")

    total_end = time.time()

    print("\n" + "=" * 60)
    print("🎉 全部任务完成！")
    print(f"⏱ 总耗时: {(total_end - total_start)/3600:.2f} 小时")
    print("=" * 60)