import numpy as np
import matplotlib.pyplot as plt

def fractal_set(xmin, xmax, ymin, ymax, width, height, max_iter):

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=complex)
    fractal = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 3 - (abs(Z[mask].real) - 1j*abs(Z[mask].imag)) ** 2 + C[mask]
        #Z[mask] = Z[mask] ** 6 - Z[mask] ** 2 + C[mask]
        fractal[mask] += 1

    return fractal

# Parameters
xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
width, height = 800, 800
max_iter = 100

# Generate the Mandelbrot set
fractal = fractal_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Plot the Mandelbrot set
plt.figure(figsize=(10, 10))
plt.imshow(fractal, extent=(xmin, xmax, ymin, ymax), cmap="hot", origin="lower")

plt.colorbar(label="Iteration count")
name = "z = $z^{6}$ - $z^{2}$ + c"
plt.title(name)
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()
