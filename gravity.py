# gravity fractal???

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

N = 3 # number of "sink holes"
r = 1.5 # radius of circle for points to be placed on
polar_angles = [360/N * k * np.pi/180 for k in range(N)]
pxs = r * np.sin(polar_angles) + np.random.random(N)
pys = r * np.cos(polar_angles) + np.random.random(N)

#pxs = np.random.random(N) * r
#pys = np.random.random(N) * r
m = 3 # xmin, xmax, ymin, ymax
res = 500 # resolution

# CONSTANTS
dt = 0.1 # small change in time
G = 6.67 # gravitational constant
A = 0.0001 # softening factor
e = 10e-2
# just dependent on N

# consider a system of N planets located at (pxs[i], pys[i]) for i in N
# now consider all points in some space and suppose a massive object was placed there
# what planet does the object fall too?
# mark each planet as a colour and plot


limit=100000
ps = np.zeros((2, limit))

# runs the gravity simulation from a starting point px, py
@njit
def run(px, py):
    dist = (px ** 2 + py ** 2) ** 1 / 2

    vx = -px / dist
    vy = -py / dist
    for i in range(limit):

        ax = 0
        ay = 0

        for j in range(N):
            dist1 = ((px - pxs[j])**2 + (py - pys[j])**2)**0.5
            if dist1 < e:
                return j
            ax -= G * (px - pxs[j]) / (dist1**2 + A**2)
            ay -= G * (py - pys[j]) / (dist1**2 + A**2)
        vx += ax * dt
        vy += ay * dt
        px += vx * dt
        py += vy * dt
    return -1


X = np.linspace(-m, m, res)
Y = np.linspace(-m, m, res)
Z = np.zeros((res, res))
for i in range(res):
    for j in range(res):
        x = X[i]
        y = Y[j]
        Z[i][j] = run(x, y)

plt.imshow(Z, extent=[-m,m,-m,m])
plt.colorbar()
plt.scatter(pxs, pys,s=1, c='k')
plt.show()
