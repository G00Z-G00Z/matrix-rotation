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
    default=150,
    help="Final angle",
)
parser.add_argument(
    "-rotation",
    type=int,
    default=10,
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
origin = np.array([0, 0], dtype=float)
rotation_matrix = rotation_deg(args.rotation)
p_0 = rotation_deg(args.initial_angle) @ np.array([a1, 0], dtype=float)
thetas = np.arange(start=args.initial_angle, stop=args.final_angle, step=args.rotation)
x_p: np.ndarray = np.empty_like(thetas)
y_p: np.ndarray = np.empty_like(thetas)


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

# Plot the robots arm on each iteration, starting from the origin, to the point formed by x_p and y_p
for i in range(thetas.size):
    plt.plot([origin[0], x_p[i]], [origin[1], y_p[i]], "k-")
    plt.plot(x_p[i], y_p[i], "ro")
    plt.plot(origin[0], origin[1], "bo")


plt.title(
    rf"Robotic arm rotation $\theta_0 = {args.initial_angle}°,\Delta\theta = {args.rotation}°, \theta_f = {args.final_angle}°$"
)
plt.xlabel(r"$x_g$")
plt.ylabel(r"$y_g$")
limit_axis = a1 * 1.5
plt.xlim((-limit_axis, limit_axis))
plt.ylim((-limit_axis, limit_axis))
plt.grid(True)
plt.savefig("cosa.png")
