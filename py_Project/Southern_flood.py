import numpy as np

Qbyd = []  # 白洋淀入淀洪水流量
Zsfy = []  # 白洋淀十方院水位
qzl = []  # 枣林庄下泄流量
Qdljh = []
z = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.2, 9.5, 10, 10.5]  # 白洋淀水位m
v = [5200, 10620, 19810, 35280, 35620, 86730, 122650, 163900, 213150, 237600, 262960, 314550, 366640]  # 白洋淀库容 万m
z_wa = []  # 文安洼水位
v_wa = []  # 文安洼库容
t = 60 * 60 * 3


def linear_interpolate(x, y, x_new):
    if x_new < x[0] or x_new > x[-1]:
        raise ValueError("Interpolation input is out of range!")
    return np.interp(x_new, x, y)


def Standard_flood(Zsfy, Vbyd):
    v_dd = 10.25 * 1e8  # 东淀设计蓄水量
    v_wa = 33.87 * 1e8  # 文安洼设计蓄水量
    vt_dd = 0  # 东淀蓄水量
    vt_wa = 0  # 文安蓄水量
    z_dlb = []  # 第六堡水位变化过程
    # 白洋淀调度过程
    for i in range(len(Zsfy) - 1):
        if 8 <= Zsfy[i] < 8.3:
            qzl.append(460)
            Vbyd -= 460 * t
        elif 8.3 <= Zsfy[i] < 9:
            qzl.append(2300)
            Vbyd -= 2300 * t
        elif 9 <= Zsfy[i] < 11.5:
            qzl.append(2700)
            Vbyd -= 2700 * t
        elif Zsfy[i] >= 11.65:
            qzl.append(2700)
            Vbyd -= (2700 + 300) * t  # 白洋淀经小关分洪闸向文安洼分洪700
        else:
            qzl.append(150)
            Vbyd -= 150 * t
        if (i + 1) < len(Zsfy):
            Zsfy[i + 1] = np.interp(Vbyd, v, z)
        else:
            Zsfy[-1] = np.interp(Vbyd, v, z)
    # 东淀运用过程
    for i in range(len(qzl)):
        if 0 <= qzl[i] < 700:
            Qdljh.append(qzl[i])
        else:
            if vt_dd < v_dd:
                Qdljh.append(qzl[i] - 200)  # 赵王新渠向东淀分洪，独流减河进洪闸来水流量减少200
                vt_dd += 200 * t
            else:
                Qdljh.append(qzl[i])
        z_dlb.append(np.interp(vt_dd, v_wa, z_wa))

    # 文安洼运用过程
    for i in range(len(Zsfy[i])):
        if Zsfy[i] >= 10.8:
            if vt_wa < v_wa:
                Qdljh[i] = Qdljh[i] - 1380  # 王村分洪闸向文安洼分洪，独流减河进洪闸来水流量减少1380
                vt_wa += 1380 * t

    # 贾口洼运用过程
    for i in range(len(z_dlb)):
        if z_dlb[i] >= 8:
            vt_wa -= vt_wa - 200  # 文安洼经锅底闸向贾口洼泄洪200
    return qzl, Qdljh, z_dlb


def Overstand_flood(Zsfy, Vbyd):
    for i in range(len(Zsfy) - 1):
        if 8 <= Zsfy[i] < 8.3:
            qzl.append(460)
            Vbyd -= 460 * t
            Qdljh.append(qzl[i])
        elif 8.3 <= Zsfy[i] < 9:
            qzl.append(2300)
            Vbyd -= 2300 * t
            Qdljh.append(qzl[i])
        elif 9 <= Zsfy[i] < 10.5:
            qzl.append(2700)
            Vbyd -= 2700 * t
            Qdljh.append(qzl[i])
        elif 10.5 <= Zsfy[i] < 11.9:
            qzl.append(2700)
            Vbyd -= 2700 * t
            Qdljh.append(qzl[i] - 1380)  # 王村分洪闸向文安洼分洪
        elif Zsfy[i] >= 11.9:
            qzl.append(2700)
            Vbyd -= (2700+300) * t
            Qdljh.append(qzl[i] - 1380 - 300)  # 小关向文安洼分洪
    return Qdljh, qzl
