import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rotation(theta) -> np.ndarray:
    theta = np.deg2rad(theta)
    return np.array(
        (
            (np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta)),
        )
    )


a1 = 0.30  # m
p_0 = np.array([a1, 0], dtype=float)
dt = 1.0

thetas = np.arange(0, 135, step=dt)

x_p = np.empty_like(thetas, dtype=float)
y_p = np.empty_like(thetas, dtype=float)
rotation_matrix = rotation(dt)

x_p[0], y_p[0] = p_0[0], p_0[1]

for i in range(1, thetas.size):
    X = rotation_matrix @ p_0
    x_p[i], y_p[i] = X[0], X[1]
    p_0 = X

df = pd.DataFrame(
    {
        "k": np.arange(0, 135),
        "theta": thetas,
        "x_p": x_p,
        "y_p": y_p,
    }
)

print(df.head())

plt.figure()
plt.plot(x_p, y_p)
plt.title("Ejercicio")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
