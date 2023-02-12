#! /bin/python
"""
    Author: Eduardo Gomez
    Title: Robotic arm movement
    Usage:
        ./rotation_matrix.py -initial_angle 90 -final_angle 420 -rotation 25"
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setting options with better display
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", "{:,.2f}".format)

# Parse user's arguments from command line
parser = argparse.ArgumentParser()
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
    """
    Returns a rotation matrix in 2 dimensions,
    that rotates a vector by an angle in deg
    """
    theta = np.deg2rad(theta)
    return np.array(
        (
            (np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta)),
        ),
        dtype=float,
    )


# Initial data
a1 = 0.30  # m
p_0 = rotation_deg(args.initial_angle) @ np.array([a1, 0], dtype=float)

# Vectors' initialization

origin = np.array([0, 0], dtype=float)
thetas = np.arange(start=args.initial_angle, stop=args.final_angle, step=args.rotation)
x_p: np.ndarray = np.empty_like(thetas)
y_p: np.ndarray = np.empty_like(thetas)

# Rotation matrix with constant rotation
rotation_matrix = rotation_deg(args.rotation)

# Assigning the first point to the vector
x_p[0], y_p[0] = p_0[0], p_0[1]

# Calculating step by step rotation
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

print("Printing all angles and positions with respect to global system")
print(df)

# Started with plotting
plt.figure()

# Plot a line from origin to point, and add a red point at the tip
for point in zip(x_p, y_p):
    plt.plot([origin[0], point[0]], [origin[1], point[1]], "k--")
    plt.plot(point[0], point[1], "ro")

# Plot origin
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
plt.savefig("img/robotic-arm.png")
