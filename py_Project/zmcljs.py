# 闸门出流计算
import math

z = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 9.85, 10, 10.5, 10.7, 11.0, 11.5, 12]  # 水位取值
q = []
m2 = 0.7
b2 = 48
C2 = 4
h = 4.8

for zt in z:
    # 1. 添加缺失的乘号 *
    # 2. 增加有效性检查
    effective_head = zt - C2 - h / 2
    if effective_head <= 0:
        print(f"警告：水位zt={zt}时，有效水头{effective_head}≤0，已跳过")
        q.append(0)  # 或者 q.append(float('nan'))
        continue

    qt = m2 * b2 * math.sqrt(2 * 9.8 * (zt - C2 - h / 2))  # 修复处：添加乘号
    q.append(qt)

print("计算流量结果(m³/s):")
for i, (zt, qt) in enumerate(zip(z, q)):
    print(f"水位 {zt}m → 流量 {qt:.2f}m³/s" if qt > 0 else f"水位 {zt}m → 无效计算")