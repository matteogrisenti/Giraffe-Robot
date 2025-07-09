from __future__ import print_function
import numpy as np
import os
import math
import pinocchio as pin
from pinocchio.utils import *
from utils.math_tools import Math
import time as tm 

import config 

def directKinematics(q):

    q1 = q[0] # shoulder_yaw joint position
    q2 = q[1] # shoulder_roll joint position
    d3 = q[2] # prismatic joint position (extension)
    q4 = q[3] # wrist_yaw joint position
    q5 = q[4] # wrist_pitch joint position

    # LOCAL homogeneous transformation matrices (base link is 0)

    # shoulder link (1)
    # rigid transform (translation along Z axis)
    T_01r = np.array([ [1, 0, 0,     0    ],
                       [0, 1, 0,     0    ],
                       [0, 0, 1, config.a1],
                       [0, 0, 0,     1    ]])
    # joint transform (rotation about Z axis)
    T_1r1 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                      [math.sin(q1), math.cos(q1),  0, 0],
                      [0,               0,              1, 0],
                      [0,               0,              0, 1]])
    # local hom. transform from link frame 0 to link frame 1
    T_01 = T_01r.dot(T_1r1)


    # arm link (2)
    # rigid transform (90deg rotation about X axis)
    T_12r = np.array([ [ 1,  0,  0, 0],
                       [ 0,  0, -1, 0],
                       [ 0, -1,  0, 0],
                       [ 0,  0,  0, 1]])
    # joint transform (rotation about Z axis)
    T_2r2 = np.array([[math.cos(q2), -math.sin(q2), 0, 0],
                      [math.sin(q2),  math.cos(q2), 0, 0],
                      [    0,             0,        1, 0],
                      [    0,             0,        0, 1]])
    # local hom. transform from link frame 1 to link frame 2
    T_12 = T_12r.dot(T_2r2)


    # extend link (3)
    # rigid transform (translation along X and Y axis, rotation about Z and X axis)
    T_23r = np.array([ [ 0,  0, 1,  config.a3_x],
                       [-1,  0, 0, -config.a3_y],
                       [ 0, -1, 0,      0      ],
                       [ 0,  0, 0,      1      ]])

    # joint transform (extension about Z axis)
    T_3r3 = np.array([ [ 1, 0, 0,  0],
                       [ 0, 1, 0,  0],
                       [ 0, 0, 1, d3],
                       [ 0, 0, 0,  1]])
    #local hom. transform from link frame 2 to link frame 3
    T_23 = T_23r.dot(T_3r3)


    # wrist link (4)
    # rigid transform (rotation about Y and X axis)
    T_34r = np.array([[ 0,  0, 1,  0],
                      [ 0, -1, 0,  0],
                      [ 1,  0, 0,  0],
                      [ 0,  0, 0,  1]])
    # joint transform  (rotation about Z axis)
    T_4r4 = np.array([[math.cos(q4), -math.sin(q4), 0, 0],
                      [math.sin(q4),  math.cos(q4), 0, 0],
                      [    0,             0,        1, 0],
                      [    0,             0,        0, 1]])
    #local hom. transform from link frame 3 to link frame 4
    T_34 = T_34r.dot(T_4r4)


    # mic link (5)
    # rigid transform (translation about Z, rotation about Y and X axis)
    T_45r = np.array([[  0,  0, 1,  0],
                      [ -1,  0, 0,  0],
                      [  0, -1, 0,  config.a5],
                      [  0,  0, 0,  1]])
    # joint transform  (rotation about Z axis)
    T_5r5 = np.array([[math.cos(q5), -math.sin(q5), 0, 0],
                      [math.sin(q5),  math.cos(q5), 0, 0],
                      [    0,             0,        1, 0],
                      [    0,             0,        0, 1]])
    #local hom. transform from link frame 3 to link frame 4
    T_45 = T_45r.dot(T_5r5)

    # mic ( end-effector )
    # only rigid transform ( translation along x)
    T_4e = np.array([[0,  0, 1,  config.a6],
                     [0, -1, 0,  0],
                     [1,  0, 0,  0],
                     [0,  0, 0,  1]])

    # GLOBAL homogeneous transformation matrices
    T_02 = T_01.dot(T_12)
    T_03 = T_02.dot(T_23)
    T_04 = T_03.dot(T_34)
    T_0e = T_04.dot(T_4e)

    return T_01, T_02, T_03, T_04, T_0e 