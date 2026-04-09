import numpy as np


def linear_interpolate(x, y, x_new):
    if x_new < x[0] or x_new > x[-1]:
        raise ValueError("Interpolation input is out of range!")
    return np.interp(x_new, x, y)


def Standard_flood(Qdc, Qbhd, Zbyd):
    qd = []  # 东茨村流量
    zd = []  # 东茨村水位
    zbh = []  # 北河店水位
    qbh = []  # 北河店流量
    qx = []  # 新盖房枢纽流量
    zx = []  # 新盖房枢纽水位
    vb = []  # 白洋淀蓄水量
    zb = []  # 白洋淀水位
    zbyd = []  # 白洋淀水位变化过程
    q_bgyh = []  # 白沟引河下泄流量
    q_fhd = []  # 新盖房分洪道下泄流量
    Q_lgw = []  # 兰沟洼分洪流量
    q_bgh = []  # 白沟河入新盖房流量
    q_njm = []  # 南拒马河入新盖房流量
    v_lgw = 3.23 * 1e8  # 兰沟洼蓄滞洪量
    t = 1 * 60 * 60  # 时间步长
    vtl = 0
    vbyd = 0
    #  东茨村标准洪水调控过程
    for i in range(len(Qdc)):
        Zdc = 0.0003 * Qdc[i] - 0.0555 * Qdc[i] + 22.0191
        if Zdc >= 28.61:
            if vtl < v_lgw:
                Q_lgw.append(Qdc[i])
                q_bgh = Qdc[i] - 100  # 白沟河向兰沟洼分洪，白沟河向新盖房枢纽入流量减少100
                vtl += 100 * t
            else:
                Q_lgw.append(0)
                q_bgh.append(Qdc[i])
        else:
            Q_lgw.append(0)
            q_bgh.append(Qdc[i])
    #  北河店标准洪水调控过程
    for i in range(len(Qbhd)):
        # Zbhd = np.interp(Qbhd[i], qbh, zbh)
        Zbhd = 0.0002 * Qbhd[i] - 0.0219 * Qbhd[i] + 20.779
        if Zbhd >= 26.3:
            if vtl < v_lgw:
                Q_lgw[i] += 100
                q_njm[i] = Qbhd[i] - 100  # 南拒马河向兰沟洼分洪，南拒马河入新盖房枢纽流量减少100
                vtl += 100 * t
            else:
                q_njm[i] = Qbhd[i]
                Q_lgw[i] += 0
        else:
            q_njm[i] = Qbhd[i]
            Q_lgw[i] += 0

    #  新盖房入流量计算
    q_bgh = np.array(q_bgh)
    q_njm = np.array(q_njm)
    if len(q_njm) == len(q_bgh):
        Qxgf = q_bgh + q_njm
    else:
        print("白沟河流量与南拒马河流量长度不匹配")
        print("len(q_njm)", q_njm)
        print("len(q_bgh)", q_bgh)

    for i in range(len(Qxgf)):
        zxgf = np.interp(Qxgf[i], qx, zx)
        if zxgf < 13.9:
            if Zbyd < 10.5:
                if Qxgf[i] > 400:
                    q_bgyh.append(400)  # 经白沟引河进入白洋淀400m3/s
                    if 8 < Zbyd < 8.3:
                        vbyd += (400 - 1840) * t
                    elif 8.3 < Zbyd < 9:
                        vbyd += (400 - 2300) * t
                    elif 9 < Zbyd < 10.5:
                        vbyd += (400 - 2700) * t
                    Zbyd = np.interp(vbyd, vb, zb)
                    zbyd.append(Zbyd)
                    q_fhd.append(Qxgf[i] - 400)
                else:
                    q_bgyh.append(Qxgf[i])
                    if 8 < Zbyd < 8.3:
                        vbyd += (Qxgf[i] - 1840) * t
                    elif 8.3 < Zbyd < 9:
                        vbyd += (Qxgf[i] - 2300) * t
                    elif 9 < Zbyd < 10.5:
                        vbyd += (Qxgf[i] - 2700) * t
                    q_fhd.append(0)
                Zbyd = np.interp(vbyd, vb, zb)  # 更新白洋淀水位
            elif Zbyd >= 10.5:
                q_bgyh.append(0)
                q_fhd.append(Qxgf[i])  # 洪水均由新盖房分洪道下泄
        else:
            q_bgyh.append(0)
            q_fhd.append(Qxgf[i])
    print("q_bgyh", q_bgyh)
    print("q_fhd", q_fhd)

    return q_bgyh, q_fhd, Qxgf, Zbyd


# 北支超标准洪水调度过程
def Overstand_flood(Qdc, Qbhd):
    qd = []  # 东茨村流量
    zd = []  # 东茨村水位
    zbh = []  # 北河店水位
    qbh = []  # 北河店流量
    qx = []  # 新盖房枢纽流量
    zx = []  # 新盖房枢纽水位
    q_bgyh = []  # 白沟引河下泄流量
    q_fhd = []  # 新盖房分洪道下泄流量
    Q_lgw = []  # 兰沟洼分洪流量
    q_bgh = []  # 白沟河入新盖房流量
    q_njm = []  # 南拒马河入新盖房流量
    v_lgw = 3.23 * 1e8  # 兰沟洼蓄滞洪量
    t = 1 * 60 * 60  # 时间步长
    vt = 0
    for i in range(len(Qdc)):
        Zdc = np.interp(Qdc[i], qd, zd)
        # 东茨村超标准洪水调度
        if Zdc >= 28.1:
            if vt < v_lgw:
                Q_lgw.append(Qdc[i])
                q_bgh[i] = Qdc[i] - 100  # 白沟河向兰沟洼分洪，新盖房枢纽来水减少100
                vt += 100 * t
            else:
                Q_lgw.append(0)
                q_bgh.append(Qdc[i])
        else:
            Q_lgw.append(0)
            q_bgh.append(Qdc[i])

    for i in range(len(Qbhd)):
        # 北河店超标准洪水调度
        Zbhd = np.interp(Qbhd[i], qbh, zbh)
        if Zbhd >= 25.9:
            if vt < v_lgw:
                Q_lgw[i] += 150
                q_bgh[i] = Qbhd[i] - 150  # 南拒马河向兰沟洼分洪，新盖房枢纽来水减少150
                vt += 150 * t
            else:
                q_bgh[i] = Qbhd[i]
                Q_lgw[i] += 0
        else:
            q_bgh[i] = Qbhd[i]
            Q_lgw[i] += 0
        z_dmy = 0.25 * vt ** 2 + 4.59 * vt
        if z_dmy >= 18.9:  # 东马营水位高于18.9米
            q_bgh[i] = Qdc[i] - 1500  # 白沟河向清北地区分洪，分洪量为1500m3/s, 即新盖房枢纽来水量减少1500
        #  新盖房入流量计算
    q_bgh = np.array(q_bgh)
    q_njm = np.array(q_njm)
    if len(q_njm) == len(q_bgh):
        Qxgf = q_bgh + q_njm
    else:
        print("白沟河流量与南拒马河流量长度不匹配")
        print("len(q_njm)", q_njm)
        print("len(q_bgh)", q_bgh)
    # 新盖房枢纽超标准洪水调度
    for i in range(len(Qxgf)):
        Zxgf = np.interp(Qxgf[i], qx, zx)
        if Zxgf < 16.7:
            q_fhd.append(Qxgf[i])  # 分洪道下泄量
            q_bgyh.append(0)  # 白沟引河下泄量
        elif Zxgf >= 16.7:  # 新盖房枢纽水位高于16.7m
            q_fhd.append(Qxgf[i] - 400)  # 新盖房分洪道向清北地区分洪，分洪道流量减少400
            q_bgyh.append(0)  # 白沟引河下泄量
    print("q_bgyh", q_bgyh)
    print("q_fhd", q_fhd)
    return q_bgyh, q_fhd, Qxgf
