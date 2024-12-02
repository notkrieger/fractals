# chaos game
import numpy as np
import matplotlib.pyplot as plt

N = 6 # number of vertices
ia = (N-2) / N * np.pi # internal angle of N-gon
a = sum([np.cos(i*(np.pi - ia)) for i in range(1, int(np.floor(N/4)) + 1)])
d = (1+2*a)/(2+2*a) # distance ratio, based on formula from chaos game wiki
d = 0.5

r = 5 # radius of circle for points to be placed on
polar_angles = [360/N * x * np.pi/180 for x in range(N)]
vxs = r * np.sin(polar_angles)
vys = r * np.cos(polar_angles)

maxIter = 100000
def chaos_game():
    lastv = -1
    px = np.zeros(maxIter)
    py = np.zeros(maxIter)
    x, y = np.random.random(), np.random.random()
    i = 0
    while i < maxIter:
        vertex = np.random.randint(N)
        if vertex == lastv:
            continue
        vx, vy = vxs[vertex], vys[vertex]
        x, y = (vx + x)*(1-d), (vy + y)*(1-d)
        px[i] = x
        py[i] = y
        i += 1
        lastv = vertex
    return px, py

px, py = chaos_game()


#plt.plot(vxs, vys, 'kx') # plot vertices
plt.plot(px,py, 'bo', ms=0.1)
plt.title(f"{N}-gon chaos game, no v twice in a row")
plt.show()
