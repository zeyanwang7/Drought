import numpy as np

def linear_interpolate(x, y, x_new):  # 线性插值
    return np.interp(x_new, x, y)


# 横山岭水库规则调度
def hsl_reservoir(Q):
    z = [231, 232, 233, 234, 235, 235.2, 236, 237, 237.5, 238, 239, 240, 241, 241.7, 242, 243, 244, 245, 245.3,
         246]  # 横山岭水库水位
    v = [0.7576, 0.8325, 0.9165, 1.0029, 1.0924, 1.107, 1.1907, 1.2904, 1.3427, 1.3949, 1.5045, 1.6184, 1.739, 1.827,
         1.8647, 1.9935, 2.1284, 2.2711, 2.3097, 2.4195]  # 横山岭水库库容
    q = [58.9, 59.8, 89.5, 142, 211, 225.2, 294, 384, 430.5, 566, 944, 1228, 1699, 2094.7, 2255, 2869, 3547, 4255,
         4488.1, 5036]  # 横山岭水库下泄流量
    t = 60 * 60 * 3
    zs = 231.13
    qs = Q[0]

    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    qx = 0
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])
        if 231.13 <= z_list[-1] < 236.63:
            if Q[i] <= qt:
                qx = qt
            elif  qt < Q[i] < 430:
                qx = Q[i]
            elif Q[i] >= 430:
                qx = 430
        elif 236.63 <= z_list[-1] < 237.73:
            if Q[i] <= qt:
                qx = qt
            elif qt < Q[i] < 800:
                qx = Q[i]
            elif Q[i] >= 800:
                qx = 800
        elif 237.73 <= z_list[-1] < 240.53:
            if Q[i] <= qt:
                qx = qt
            elif  qt < Q[i] < 1900:
                qx = Q[i]
            elif Q[i] >= 1900:
                qx = 1900
        elif z_list[-1] >= 240.53:
                qx = 1900
        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qx) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qx, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list


def lm_reservoir(Q):
    z = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131]
    v = [0.3365, 0.3929, 0.4546, 0.521, 0.5922, 0.6696, 0.7541, 0.8558, 0.9439, 1.0472, 1.1566, 1.272]
    q = [0, 119, 371, 725, 1164, 1681, 2346, 3153, 4073, 5106, 6249, 7503]
    zs = 119.97
    qs = Q[0]
    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    qx = 0
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])
        if z_list[-1] < 124.39:
            if qt > 300:
                qt = 300
        elif 124.39<= z_list[-1] < 127.02:
            qt = 450
        elif z_list[-1] >= 127.02:
            qt = Q[i]

        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qt, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list


def wk_reservoir(Q):
    z = [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
         214, 215]
    v = [4.03, 4.329, 4.644, 4.993, 5.301, 5.673, 6.049, 6.434, 6.834, 7.241, 7.66, 8.097, 8.575, 9.057, 9.549, 10.05,
         10.57, 11.136, 11.708, 12.288, 12.879, 13.484, 14.134, 14.789]
    q = [238, 451, 860, 1401, 2047, 2785, 3600, 4487, 5361, 6221, 7115, 8094, 9112, 10172, 11270, 12405, 13576, 14782,
         16020, 17294, 18599, 19936, 21303, 22701]
    zs = 191.24
    qs = Q[0]
    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    qx = 0
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])
        if 191.24 <= z_list[-1] < 197.53:
            if Q[i] <= qt:
                qx = qt
            elif  qt < Q[i] < 800:
                qx = Q[i]
            elif Q[i] >= 800:
                qx = 800
            if qx > 800:
                qx = 800
        elif 197.53 <= z_list[-1] < 199.37:
            if  qt <  3000:
                qx = qt
            elif qt >= 3000:
                qx = 3000
            if qx > 3000:
                qx = 3000
        elif 199.37 <= z_list[-1] < 202.8:
            if Q[i] <= qt:
                qx = qt
            elif Q[i] > qt:
                if Q[i] > 7000:
                    qx = 7000
                else:
                    qx = qt
            if qx > 7000:
                qx = 7000
        elif z_list[-1] >= 202.8:
            qx = qt

        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qx) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qx, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list


def xdy_reservoir(Q):
    z = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
         152, 153]  # 西大洋水库水位#
    v = [2.404, 2.641, 2.895, 3.176, 3.463, 3.765, 4.081, 4.432, 4.795, 5.167, 5.564, 5.972, 6.412, 6.874, 7.353, 7.836,
         8.331, 8.859, 9.407, 9.965, 10.531, 11.139, 11.746, 12.417]  # 西大洋水库水位对应库容
    q = [185, 188, 252, 463, 795, 1211, 1696, 2235, 2822, 3576, 4471, 5464, 6569, 7795, 9070, 10364, 11715, 13125,
         14592, 16113, 17687, 19311, 20986, 22707]
    zs = 130.85
    qs = Q[0]

    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]

    # print(vs)
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])
        qx = qt
        if 130.85 <= z_list[-1] < 137.66:
            if qt > 300:
                qx = 300
        elif 137.66 <= z_list[-1] < 139.97:
            if qt > 1000:
                qx = 1000
        elif 139.97 <= z_list[-1] < 142.33:
            if qt > 5460:
                qx = 5400

        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qx) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qx, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list


def has_positive(numbers):
    return any(x > 0 for x in numbers)


def fhl_js(qfh, fhmax):
    w = 0
    for i in range(len(qfh)-1):
        w = w + (qfh[i] + qfh[i + 1]) * 3 * 60 * 30 / 1e8
    if w > fhmax:
        return False
    else:
        return True


def byd(Q):  # 白洋淀调控过程
    zs = 6.5
    q = [42, 200, 494, 1260, 1990, 2700, 4260, 4415]
    z = [5, 6, 7, 8, 8.5, 9, 10.5, 11]
    v = [0.52, 1.71, 4.197, 8.185, 10.61, 13.58, 22.03, 24]
    qs = 42
    qzw = [0]
    qwa = [0]
    vs = np.interp(zs, z, v)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    for i in range(len(Q) - 1):
        qt = linear_interpolate(z, q, z_list[-1])  # 最大下泄能力
        if 9 < z_list[-1] < 9.2:
            v9 = linear_interpolate(z, v, 9)
            vt = linear_interpolate(z, v, z_list[-1])
            qs = (vt - v9) * 100000000 / 3 * 60 * 60  # 水位降至9m原需要下泄流量
            if qt <= Q[i] + qs < qt + 11823.98:
                qzwt = Q[i] + qs - qt
                if qt > 1380:
                    qwat = 1380
                else:
                    qwat = qt
            elif qt + 11823.9 <= Q[i] + qs:
                qzwt = 11823.98
                if qt > 1380:
                    qwat = 1380
                else:
                    qwat = qt

            elif 0 <= Q[i] + qs < qt:
                qzwt = 0
                qt = Q[i] + qs
                if qt > 1380:
                    qwat = 1380
                else:
                    qwat = qt
            if fhl_js(qzw, 22.8):
                qzw.append(qzwt)
                qwa.append(qwat)
            else:
                qzw.append(0)
                qwa.append(qzwt+qwat)

        elif z_list[-1] >= 9.2:
            v9 = linear_interpolate(z, v, 9)
            vt = linear_interpolate(z, v, z_list[-1])
            qs = (vt - v9) * 100000000 / 3 * 60 * 60
            if qt <= Q[i] + qs < qt + 11823.98:
                if qt > 1380:
                    qwat = 5280
                else:
                    qwat = qt + 3900
                qzwt = Q[i] + qs - qt - qwat
            elif qt + 11823.9 <= Q[i] + qs < qt + 11823.98 + 5280:
                qzwt = 11823.98
                if qt > 1380:
                    qwat = 5280
                else:
                    qwat = qt + 3900
            elif qt + 11823.98 + 5280 <= Q[i] + qs:
                qzwt = 11823.98
                if qt > 1380:
                    qwat = 5280
                else:
                    qwat = qt + 3900
            elif 0 <= Q[i] + qs < qt:
                qt = Q[i] + qs
                if qt > 1380:
                    qwat = 5280
                else:
                    qwat = qt + 3900
            if fhl_js(qzw, 22.8):
                qzw.append(qzwt)
                qwa.append(qwat)
            else:
                qzw.append(0)
                qwa.append(qwat+qzwt)
        else:
            qzwt = 0
            qwat = 0
            if has_positive(qzw):
                if 0 < Q[i] - qt < 11823.98:
                    qzwt = Q[i] - qt
                elif Q[i] - qt > 11823.98:
                    qzwt = 11823.98
                elif Q[i] - qt < 0:
                    qzwt = 0
            if has_positive(qwa):
                if qt >= 1380:
                    qwat = 1380
                else:
                    qwat = qt

            if fhl_js(qzw, 22.8):
                qzw.append(qzwt)
                qwa.append(qwat)
            else:
                qzw.append(0)
                qwa.append(qzwt + qwat)
        vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[
            -1] * 100000000 - (qzw[-1] + qzwt) * 3 * 60 * 60 / 2 - (qwa[-1] + qwat) * 3 * 60 * 60 / 2
        v2 = vt / 100000000
        zt = linear_interpolate(v, z, v2)
        q_list.append(round(qt, 2))
        v_list.append(round(v2, 2))
        z_list.append(round(zt, 2))
    return z_list, q_list, qzw, qwa


def angezhuang_reservoir(Q, lj):
    z = [143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
         164, 165, 166, 167, 168, 169]
    v = [0.3493, 0.3995, 0.4527, 0.513, 0.574, 0.6368, 0.7049, 0.7846, 0.8669, 0.9475, 1.0271, 1.1177, 1.2157, 1.3158,
         1.4201, 1.5283, 1.6385, 1.7562, 1.8745, 2.0023, 2.131, 2.2608, 2.3985, 2.546, 2.696, 2.8485, 3.0075]
    q = [160, 164, 169, 173, 177, 182, 186, 189, 192, 195, 420, 556, 714, 889, 1082, 1290, 1514, 1752, 2005, 2273,
         2555, 2851, 3162, 3487, 3828, 4182, 4552]
    zs = 150.74
    qs = Q[0]

    vs = round(linear_interpolate(z, v, zs), 3)  # 起调库容
    v_list = [vs]
    z_list = [zs]
    q_list = [qs]
    print(vs)
    if lj == 0:
        for i in range(len(Q) - 1):
            qt = linear_interpolate(z, q, z_list[-1])
            if 150.74 <= z_list[-1] < 153.90:
                if qt > 300:
                    qt = 300
            if 153.90 <= z_list[-1] < 155.58:
                if qt > 1000:
                    qt = 1000
            if 155.58 <= z_list[-1] < 158.16:
                if qt > 1598.9:
                    qt = 1598.9

            vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
            v2 = vt / 100000000
            zt = linear_interpolate(v, z, v2)
            q_list.append(round(qt, 2))
            v_list.append(round(v2, 2))
            z_list.append(round(zt, 2))
    else:  # 超标准洪水>100年
        for i in range(len(Q) - 1):
            qt = linear_interpolate(z, q, z_list[-1])
            if qt > Q[i + 1]:
                qt = Q[i + 1]
            vt = (Q[i] + Q[i + 1]) * 3 * 60 * 60 / 2 - (q_list[-1] + qt) * 3 * 60 * 60 / 2 + v_list[-1] * 100000000
            v2 = vt / 100000000
            zt = linear_interpolate(v, z, v2)
            q_list.append(round(qt, 2))
            v_list.append(round(v2, 2))
            z_list.append(round(zt, 2))
    return z_list, q_list


def muskingum_non_linear(inflow, K, x, dt, O0, n):
    """
    非线性马斯京根法计算流量演算。

    参数：
    inflow: list or numpy array，入流量序列。
    K: float，马斯京根储存参数。
    x: float，权重系数，范围为0到0.5。
    dt: float，时间步长。
    O0: float，初始出流量。
    n: float，非线性指数。

    返回：
    outflow: numpy array，出流量序列。
    """
    # 参数计算
    C0 = (dt - 2 * K * x) / (2 * K * (1 - x) + dt)
    C1 = (dt + 2 * K * x) / (2 * K * (1 - x) + dt)
    C2 = (2 * K * (1 - x) - dt) / (2 * K * (1 - x) + dt)
    print("C0", C0)
    print("C1", C1)
    print("C2", C2)
    # 初始化出流量序列
    outflow = [O0]

    # 逐时间步计算出流量
    for t in range(1, len(inflow)):
        # 非线性修正
        O_prev = outflow[-1]
        O_t = (C0 * inflow[t] + C1 * inflow[t - 1] + C2 * O_prev) ** n
        outflow.append(O_t)

    return np.array(outflow)
