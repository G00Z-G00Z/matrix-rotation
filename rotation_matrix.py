import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    default=360,
    help="Final angle",
)
parser.add_argument(
    "-rotation",
    type=int,
    default=1,
    help="Constant rotation to rotate by",
)
args = parser.parse_args()


def rotation_deg(theta) -> np.ndarray:
    theta = np.deg2rad(theta)
    return np.array(
        (
            (np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta)),
        ),
        dtype=float,
    )


a1 = 0.30  # m
p_0 = rotation_deg(args.initial_angle) @ np.array([a1, 0], dtype=float)
dt: int = args.rotation

thetas = np.arange(
    start=args.initial_angle, stop=args.final_angle, step=dt, dtype=float
)

x_p: np.ndarray = np.empty_like(thetas)
y_p: np.ndarray = np.empty_like(thetas)

rotation_matrix = rotation_deg(dt)

x_p[0], y_p[0] = p_0[0], p_0[1]

for i in range(1, thetas.size):
    X = rotation_matrix @ p_0
    x_p[i], y_p[i] = X[0], X[1]
    p_0 = X


df = pd.DataFrame(
    {
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
