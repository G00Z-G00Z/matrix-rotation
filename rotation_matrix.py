# Import standard libraries
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Parse user's arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "-amplitude", type=float, default=1.0, help="Amplitude of function sin()"
)
parser.add_argument(
    "-initial_angle",
    type=float,
    default=0.0,
    help="Initial angle",
)
parser.add_argument(
    "-final_angle",
    type=float,
    default=6.28,
    help="Final angle",
)
parser.add_argument(
    "-rotation",
    type=int,
    default=1,
    help="Constant rotation to rotate by",
)
args = parser.parse_args()


# Initialise a numpy-type list
theta = np.arange(start=args.initial_angle, stop=args.final_angle, step=args.rotation)

# Evaluate the function 'sin' on 'x'
y = args.amplitude * np.sin(theta)

# Visualise the function 'y = sin(x)'
plt.figure(1)
plt.plot(theta, y, linewidth=2)
plt.title(r"Function $y=\sin(\theta)$ ")
plt.xlabel(r"$\theta$")
plt.ylabel("y")
plt.show()
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
