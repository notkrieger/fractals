# gravity fractal???
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit
from matplotlib import colors, cm

N = 4 # number of "sink holes"
r = 1.5 # radius of circle for points to be placed on
polar_angles = [360/N * k * np.pi/180 for k in range(N)]
print(polar_angles)
pxs = r * np.sin(polar_angles) #+ np.random.normal(size=N) / 7
pys = r * np.cos(polar_angles) #+ np.random.normal(size=N) / 7

res = 500 # resolution

# CONSTANTS
dt = 0.075 # small change in time
G = 6.67 # gravitational constant
A = 0.001 # softening factor
e = 0.2

colours = ['b', 'r', 'g', 'c', 'm', 'y', 'indigo']
cmap = colors.ListedColormap(colours[:N+1])
cmap = cm.get_cmap('Spectral')

# consider a system of N planets located at (pxs[i], pys[i]) for i in N
# now consider all points in some space and suppose a massive object was placed there
# what planet does the object fall too?
# mark each planet as a colour and plot


limit=10001//2
ps = np.zeros((2, limit))

# runs the gravity simulation from a starting point px, py
@njit
def run(px, py):
    speed = (px ** 2 + py ** 2) ** 1 / 2

    vx = + np.random.normal()/100 #randomise inital speed
    vy = + np.random.normal()/100
    vx = vy = 0 # make initial speed 0
    closest_p = -1
    p_dist = 10000

    for i in range(limit):

        ax = 0
        ay = 0


        for j in range(N):
            dist1 = ((px - pxs[j])**2 + (py - pys[j])**2)**0.5
            if dist1 < p_dist:
                p_dist = dist1
                closest_p = j
            if dist1 < e:
                return j
            ax -= G * (px - pxs[j]) / (dist1**2 + A**2)
            ay -= G * (py - pys[j]) / (dist1**2 + A**2)
        vx += ax * dt
        vy += ay * dt
        px += vx * dt
        py += vy * dt
    return closest_p


@njit
def singleParticle(ix, iy):
    s_pxs = np.zeros(limit)
    s_pys = np.zeros(limit)

    s_pxs[0] = ix
    s_pys[0] = iy

    vx = vy = 0

    for i in range(1, limit):
        px = s_pxs[i-1]
        py = s_pys[i-1]

        ax = 0
        ay = 0


        for j in range(N):
            dist1 = ((px - pxs[j])**2 + (py - pys[j])**2)**0.5

            ax -= G * (px - pxs[j]) / (dist1**2 + A**2)
            ay -= G * (py - pys[j]) / (dist1**2 + A**2)
        vx += ax * dt
        vy += ay * dt
        px += vx * dt
        py += vy * dt
        s_pxs[i] = px
        s_pys[i] = py

    return s_pxs ,s_pys


m = 2000 # xmin, xmax, ymin, ymax
zoom = False
zoom_x = 2
zoom_y = 2
offset = 0.5
singleParticleDemo = False

if singleParticleDemo:
    # single particle path
    s_pxs, s_pys = singleParticle(1.1111, 1.21312)
    plt.plot(s_pxs, s_pys, 'b-', ms=0.5)
    for i in range(N):
        plt.scatter(pys[i], pxs[i], s=1.5, c='k')
    plt.show()
else:
    if zoom:
        X = np.linspace(zoom_x-offset, zoom_x+offset, res)
        Y = np.linspace(zoom_y-offset, zoom_y+offset, res)
    else:
        X = np.linspace(-m, m, res)
        Y = np.linspace(-m, m, res)
    Z = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x = X[i]
            y = Y[j]
            Z[i][j] = run(x, y)



    plt.figure(figsize=(8, 6))
    if zoom:
        plt.imshow(Z, extent=[zoom_x-offset, zoom_x+offset, zoom_y-offset, zoom_y+offset], cmap=cmap)
    else:
        plt.imshow(Z, extent=[-m,m,-m,m], cmap=cmap)
    plt.colorbar(ticks=np.linspace(0, N, N+1))
    for i in range(N):
        if zoom:
            break
        plt.scatter(pys[i], pxs[i], s =1.5, c='k')
    plt.show()


