# lyapunov fractal
import numpy as np
import matplotlib.pyplot as plt


S = "ABBAAABBAB" # sequence which we apply the different r's

res = 250 # resolution
lowx, highx = 1, 4
lowy, highy = 1, 4
x = np.linspace(lowx, highx, res)
y = np.linspace(lowy, highy, res)
X, Y = np.meshgrid(x, y)

fractal = np.zeros_like(X)

max_iter = 10**4 // 2 # number of terms to solve lyapunov exponent
initVal = 0.5

def calc_exponent(a, b, S):
    # series of values gained from applying logistic map repeadtely
    # assume series is max_iter long
    seqLen = len(S)
    val = 0
    z = initVal
    for ii in range(max_iter):
        rs = S[ii % seqLen]
        if rs == 'A':
            r = a
        else:
            r = b
        if ii > 1:
            o = np.log(abs(r*(1-2*z)))
            val += o
        z = r * z * (1 - z)
    return val/(max_iter - 1)


for i in range(res):
    print(i)
    for j in range(res):
        exp = calc_exponent(x[i], y[j], S)
        if not np.isinf(exp):
            fractal[res-1-i, j] = exp
        else:
            fractal[res-1-i, j] = 0

plt.imshow(fractal, extent=[lowx, highx, lowy, highy], cmap='PuOr_r')
plt.title("Lyapunov fractal from: " + S)
plt.show()
