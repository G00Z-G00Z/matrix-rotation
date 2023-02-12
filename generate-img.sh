#!/bin/bash

./rotation_matrix.py -filename robotic-arm.png
./rotation_matrix.py -initial_angle 0 -final_angle 30 -rotation 3 -filename rotation1.png
./rotation_matrix.py -initial_angle 30 -final_angle 110 -rotation 10 -filename rotation2.png
./rotation_matrix.py -initial_angle 90 -final_angle 200 -rotation 20 -filename rotation3.png
