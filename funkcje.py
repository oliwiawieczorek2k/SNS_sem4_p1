import math
import numpy
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc, rcParams, grid
import matplotlib.patches as mpatches
from pylab import *


#odczytanie pliku
def get_data(file_name):
    file = open(file_name)
    almanac = []
    for i in range(31):
        satellite = []
        file.readline()
        for j in range(13):
            line = file.readline()
            satellite.append(line[line.index(":") + 1:].strip())
        file.readline()
        almanac.append(satellite)
    file.close()
    return almanac


#zamiana czasu
def gps_time(ddate):
    dday = date.toordinal(date(ddate[0], ddate[1], ddate[2])) - date.toordinal(date(1980, 1, 6))
    t = ((dday // 7) - 2048) * 604800 + (dday % 7) * 86400 + ddate[3] * 3600 + ddate[4] * 60
    return t


#wyznaczenie pozycji satelity
def satellite_position(t, i):
    mi = 3.986004415 * pow(10, 14)
    omega_e = 7.2921151467 / pow(10, 5)

    e = float(almanac[i][2])
    toa= float(almanac[i][3])
    inc= float(almanac[i][4])
    omega_dot= float(almanac[i][5])
    sqrt_a= float(almanac[i][6])
    omega0= float(almanac[i][7])
    omega= float(almanac[i][8])
    mo= float(almanac[i][9])
    week= int(almanac[i][12])

    pos = []

    tk = t - (week * 604800 + toa)
    n = math.sqrt(mi) / pow(sqrt_a, 3)
    mk = (mo + (n*tk)) % (2 * math.pi)
    ek = mk
    e0 = 0
    while abs(ek - e0) > pow(10, -15):
        e0 = ek
        ek = mk + e * math.sin(e0)
    vk = math.atan2((math.sqrt(1 - pow(e, 2)) * math.sin(ek)),
                    (math.cos(ek) - e))
    fik = vk + omega
    rk = pow(sqrt_a, 2) * (1 - e * math.cos(ek))
    xk = rk * math.cos(fik)
    yk = rk * math.sin(fik)
    oma = omega0 + ((omega_dot - omega_e) * tk) - (omega_e * toa)

    x = xk * math.cos(oma) - yk * math.cos(inc) * math.sin(oma)
    y = xk * math.sin(oma) + yk * math.cos(inc) * math.cos(oma)
    z = yk * math.sin(inc)

    pos.append(x)
    pos.append(y)
    pos.append(z)

    return pos


#wyznaczenie azymutu i elewacji
def vectors(t,fi, lambd, h, i):
    pos = satellite_position(t,i)

    e2 = 0.00669438002290
    a = 6378137

    n = a / (math.sqrt(1 - e2 * pow(math.sin(fi), 2)))
    xr = (n + h) * math.cos(fi) * math.cos(lambd)
    yr = (n + h) * math.cos(fi) * math.sin(lambd)
    zr = (n * (1 - e2) + h) * math.sin(fi)

    xs = pos[0]
    ys = pos[1]
    zs = pos[2]

    xsr = numpy.array([[xs - xr], [ys - yr], [zs - zr]])
    neu = numpy.array([[-math.sin(fi) * math.cos(lambd), -math.sin(lambd), math.cos(fi) * math.cos(lambd)],
                       [-math.sin(fi) * math.sin(lambd), math.cos(lambd), math.cos(fi) * math.sin(lambd)],
                       [math.cos(fi), 0, math.sin(fi)]])
    xsr_neu = numpy.dot(neu.T, xsr)

    az = numpy.rad2deg(math.atan2(xsr_neu[1], xsr_neu[0]))
    if az < 0:
        az = az + 360
    el = numpy.rad2deg(math.asin((xsr_neu[2]) / math.sqrt(xsr_neu[0] ** 2 + xsr_neu[1] ** 2 + xsr_neu[2] ** 2)))

    return az,el


#wyznacznie wspolczynnikow DOP
def dop(t,fi, lambd, h,mask):
    e2 = 0.00669438002290
    a = 6378137

    n = a / (math.sqrt(1 - e2 * pow(math.sin(fi), 2)))
    xr = (n + h) * math.cos(fi) * math.cos(lambd)
    yr = (n + h) * math.cos(fi) * math.sin(lambd)
    zr = (n * (1 - e2) + h) * math.sin(fi)

    A = numpy.array([[1, 1, 1, 1]])
    for i in range(31):
        el = vectors(t, fi, lambd, h, i)[1]
        if el > mask:
            xs = satellite_position(t, i)[0]
            ys = satellite_position(t, i)[1]
            zs = satellite_position(t, i)[2]
            ro = math.sqrt(pow((xs-xr), 2) + pow((ys-yr), 2) + pow((zs-zr), 2))
            A = numpy.append(A, [[- (xs - xr) / ro,  - (ys - yr) / ro,  - (zs - zr) / ro, 1]], axis=0)

    A = numpy.delete(A, 0, axis=0)
    Q = numpy.dot(A.T, A)
    Q = numpy.linalg.inv(Q)

    GDOP = math.sqrt(Q[0][0] + Q[1][1] + Q[2][2] + Q[3][3])
    PDOP = math.sqrt(Q[0][0] + Q[1][1] + Q[2][2])
    TDOP = math.sqrt(Q[3][3])

    Rneu = numpy.array([[-math.sin(fi) * math.cos(lambd), -math.sin(lambd), math.cos(fi) * math.cos(lambd)],
                         [-math.sin(fi) * math.sin(lambd), math.cos(lambd), math.cos(fi) * math.sin(lambd)],
                         [math.cos(fi), 0, math.sin(fi)]])
    Qxyz = numpy.delete(Q, 3, axis=1)
    Qxyz = numpy.delete(Qxyz, 3, axis=0)
    Qneu = numpy.dot(Rneu.T, Qxyz)
    Qneu = numpy.dot(Qneu, Rneu)

    HDOP = math.sqrt(Qneu[0][0] + Qneu[1][1])
    VDOP = math.sqrt(Qneu[2][2])
    PDOPneu = math.sqrt(Qneu[0][0] + Qneu[1][1] + Qneu[2][2])

    return [GDOP,PDOP,TDOP,HDOP,VDOP,PDOPneu]


#rysowanie wykresu elewacji
def plot_el(d_start, d_end, dt, nr_sat, mask):
    t0 = d_start

    if nr_sat == "all":
        nr_sat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    for n in nr_sat:
        if n > 0 and n <= 10:
            i = n - 1
        elif n >= 12 and n <= 32:
            i = n - 2

        elev = []
        time = []
        d_start = t0
        while d_start <= d_end:
            tmp = d_start/3600-16656
            if tmp >= 24:
                tmp = tmp - 24
            time.append(tmp)
            elev.append(vectors(d_start,fi,lambd,h,i)[1])
            d_start = d_start + dt
        plt.plot(time, elev,label=str(n))

    plt.legend(frameon=False, loc='upper center', ncol=6)
    plt.xlabel("Time GPS")
    plt.ylabel("Elevation")

    plt.axis([time[0], time[-1], mask, 140])
    plt.show()


#rysowanie wykresu przedstawiającego ilosc widocznych satelit
def nr_visible(d_start,d_end,dt,mask):
    t0 = d_start

    count = []
    time = []
    d_start = t0
    while d_start <= d_end:
        cnt = 0
        tmp = d_start / 3600 - 16656
        if tmp >= 24:
            tmp = tmp - 24
        time.append(tmp)
        for i in range(31):
            if vectors(d_start,fi,lambd,h,i)[1] > mask:
                cnt = cnt+1
        count.append(cnt)
        d_start = d_start + dt
    plt.plot(time,count)
    plt.axis([time[0],time[-1], 0, 32])
    plt.xlabel("Time GPS")
    plt.ylabel("Number of satellites visible")
    plt.show()


#rysowanie wykresu wspolczynnikow DOP
def plot_dop(d_start,d_end,dt,fi,lambd,h,mask):
    time = []

    gdop = []
    pdop = []
    tdop = []
    hdop = []
    vdop = []
    pdop_neu = []

    while d_start <= d_end:
        tmp = d_start / 3600 - 16656
        if tmp >= 24:
            tmp = tmp - 24
        time.append(tmp)

        gdop.append(dop(d_start, fi, lambd, h,mask)[0])
        pdop.append(dop(d_start, fi, lambd, h,mask)[1])
        tdop.append(dop(d_start, fi, lambd, h,mask)[2])
        hdop.append(dop(d_start, fi, lambd, h,mask)[3])
        vdop.append(dop(d_start, fi, lambd, h,mask)[4])
        pdop_neu.append(dop(d_start, fi, lambd, h,mask)[5])

        d_start = d_start + dt

    plt.plot(time,gdop,label='GDOP')
    plt.plot(time,pdop,label='PDOP')
    plt.plot(time,tdop,label='TDOP')
    plt.plot(time,hdop,label='HDOP')
    plt.plot(time,vdop,label='VDOP')
    plt.plot(time,pdop_neu,label='PDOPneu')
    plt.legend()
    plt.axis([time[0],time[-1], 0, 6])
    plt.xlabel("Time GPS")
    plt.ylabel("DOP values")
    plt.show()


#rysowanie skyplot
def plot_skyplot(t,fi,lambd,h,mask):
    rc('grid', color='gray', linewidth=1, linestyle='--')
    fontsize = 20
    rc('xtick', labelsize=fontsize)
    rc('ytick', labelsize=fontsize)
    rc('font', size=fontsize)


    # start ploting
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.1,
                        top=0.85,
                        left=0.1,
                        right=0.74)
    ax = fig.add_subplot(polar=True)  # define a polar type of coordinates
    ax.set_theta_zero_location('N')  # ustawienie kierunku północy na górze wykresu
    ax.set_theta_direction(-1)  # ustawienie kierunku przyrostu azymutu w prawo
    PG = 0

    azmaska = numpy.arange(0, 2 * numpy.pi, 0.01)
    elmaska = mask * numpy.ones((azmaska.shape[0]))
    ax.plot(azmaska, 90 - elmaska, color=(0.4, 0.4, 0.4))

    for i in range(31):
        az = vectors(t,fi,lambd,h,i)[0]
        el = vectors(t, fi, lambd, h, i)[1]
        PRN = almanac[i][0]

        if el > mask:
            PG += 1
            ax.annotate("G"+str(PRN),
                        xy=(numpy.deg2rad(az), 90 - el),
                        bbox=dict(boxstyle="round", fc='#7A68A6', alpha=0.5),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        color='k')

    gps = mpatches.Patch(color='#7A68A6', label='{:02.0f}  GPS'.format(PG))
    plt.legend(handles=[gps], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # axis ticks descriptions
    ax.set_yticks(range(0, 90 + 10, 10))  # Define the yticks
    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    ax.set_yticklabels(yLabel)
    # saving and showing plot
    # plt.savefig('satellite_skyplot.pdf')
    plt.show()  # wyświetleni