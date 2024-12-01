# newton fractal generator
import numpy as np
import matplotlib.pyplot as plt



# main method
max_iter = 100 # more iterations may be required for larger order polynomials
res = 1000
low, high = 0.5, 1.5
x = np.linspace(low, high, res)
y = np.linspace(low, high, res)

X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
fractal = np.zeros_like(Z, dtype='int') # list of ints showing which root a point goes to
iters = np.zeros_like(Z, dtype='float') # percentage of max_iters before convergance


# line to change to generate new fractals
# can handle complex numbers but only use imaginary part for working graph titles
# a+bi doesnt appear to make much of a difference from bi anyway??? -- no idea if this is correct
p = [-16, 0, 0, 0, 15, 0, 0, 0, 1] # p[0] + p[1] * x + p[2] * x^2 + ...


poly = np.polynomial.Polynomial(p)
eps = 0.000001

for i in range(max_iter):
    Z -= poly(Z)/poly.deriv(1)(Z) # newton method step
    for j, root in enumerate(poly.roots()):
        mask = np.abs(Z - root) < eps
        fractal[mask] = j
        iters[mask] = i / max_iter
        Z[mask] = np.nan # stop searching for solved

# make polynomial equation pretty for graph title
def custom_poly_str(coeffs):
    str1 = "f(x) = "
    order = len(coeffs)
    for ind, i in enumerate(reversed(coeffs)):
        order -= 1
        if i == 0:
            continue
        si = str(i)
        if abs(np.imag(i)) > 0:
            if si[0] == '(': # negative number
                si = si[3:-1]

        if order == len(coeffs) - 1:
            str1 += si
            str1 += f"$x^{order}$"
            continue
        else:
            if si[0] != '-':
                si = "+" + si
            if np.imag(i) == 0 and order != 0:
                print(si)
                if abs(i) == 1:
                    si = si.replace('1', '')
            str1 += si
            if order > 1:
                str1 += f"$x^{order}$"
            if order == 1:
                str1 += 'x'
    return str1

def hsv_to_rgb(col):
    h, s, v = col
    h = h % 360  # Ensure Hue is in [0, 360)
    c = v * s  # Chroma
    x = c * (1 - abs((h / 60) % 2 - 1))  # Intermediate value
    m = v - c  # Adjustment factor

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:  # 300 <= h < 360
        r, g, b = c, 0, x

    # Add the adjustment factor and return scaled RGB values
    return (r + m, g + m, b + m)


# calculates hsv colour value based on iterations and root
def colour(fractal, iters):
    HSV = np.zeros((res, res, 3), dtype='float')
    hues = 360 / (len(p) - 1) # number of hues needed (aka num roots)
    for ii in range(res):
        for jj in range(res):
            HSV[ii][jj][0] = fractal[ii][jj] * hues
            HSV[ii][jj][1] = iters[ii][jj] * 0.5 + 0.5 # some factor and offset to make colours better
            HSV[ii][jj][2] = 1 - iters[ii][jj]

    return HSV


HSV = colour(fractal, iters)

final = np.zeros((res, res, 3), dtype='float')
# convert hsv to rgb
for i in range(res):
    for j in range(res):

        col = hsv_to_rgb(HSV[i][j])
        final[i][j][0] = col[0]
        final[i][j][1] = col[1]
        final[i][j][2] = col[2]

# plot fractal
plt.imshow(final, cmap='viridis', extent = [low, high, low, high])
plt.title("Newton's Fractal of: {}".format(custom_poly_str(p)))
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()




