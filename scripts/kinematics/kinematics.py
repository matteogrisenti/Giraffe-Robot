from __future__ import print_function
import numpy as np
import os
import math
import pinocchio as pin
from pinocchio.utils import *
import time as tm 



# configuration parameters for the Giraffe robot URDF model
shoulder_joint_radius = 0.05    # shoulder joint radius
shoulder_joint_height = 0.1     # shoulder joint height
arm_length = 2.4                # arm length
arm_radius = 0.05               # arm radius
prismatic_height_delta = 0.1    # prismatic extension height delta
wrist_joint_height = 0.1        # wrist joint height
wrist_radius = 0.025            # wrist joint radius
wrist_link_length = 1.0         # wrist link length
mic_stick_length = 0.15         # microphone stick length
mic_stick_radius = 0.015        # microphone stick radius

a1 = shoulder_joint_height
a3_x = shoulder_joint_radius + arm_length + wrist_radius
a3_y = prismatic_height_delta
a5 = wrist_link_length - wrist_radius
a6 = wrist_radius + mic_stick_length + 2*mic_stick_radius



def directKinematics(q):

    q1 = q[0] # shoulder_yaw joint position
    q2 = q[1] # shoulder_roll joint position
    d3 = q[2] # prismatic joint position (extension)
    q4 = q[3] # wrist_yaw joint position
    q5 = q[4] # wrist_pitch joint position 

    # LOCAL homogeneous transformation matrices (base link is 0)
    
    # base link (0) - rigid transformation from the world frame to the base frame
    # rigid transformation from the world frame to the floor frame
    T_wf = np.array([ [1, 0, 0,  2.5],
                      [0, 1, 0,  6.0],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
    
    # rigid transformation from the floor frame to the ceiling frame
    T_fc = np.array([ [1, 0, 0,  0],
                      [0, 1, 0,  0],
                      [0, 0, 1,  4],
                      [0, 0, 0,  1]])
    
    # rigid transformation from the ceiling frame to the base frame
    T_c0 = np.array([ [ 0, -1,  0,  0],
                      [-1,  0,  0,  0],
                      [ 0,  0, -1,  0],
                      [ 0,  0,  0,  1]])
    
    T_w0 = T_wf.dot(T_fc).dot(T_c0)


    # shoulder link (1)
    # rigid transform (translation along Z axis)
    T_01r = np.array([ [1, 0, 0,  0],
                       [0, 1, 0,  0],
                       [0, 0, 1, a1],
                       [0, 0, 0,  1]])
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
                       [ 0,  0,  1, 0],
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
    T_23r = np.array([ [ 0,  0, 1,  a3_x],
                       [-1,  0, 0, -a3_y],
                       [ 0, -1, 0,    0 ],
                       [ 0,  0, 0,    1 ]])

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
    T_45r = np.array([[  0,  0, 1,  0 ],
                      [ -1,  0, 0,  0 ],
                      [  0, -1, 0,  a5],
                      [  0,  0, 0,  1 ]])
    # joint transform  (rotation about Z axis)
    T_5r5 = np.array([[math.cos(q5), -math.sin(q5), 0, 0],
                      [math.sin(q5),  math.cos(q5), 0, 0],
                      [    0,             0,        1, 0],
                      [    0,             0,        0, 1]])
    #local hom. transform from link frame 3 to link frame 4
    T_45 = T_45r.dot(T_5r5)

    # mic ( end-effector )
    # only rigid transform ( translation along x)
    T_5e = np.array([[1,  0, 0,  a6],
                     [0,  1, 0,  0 ],
                     [0,  0, 1,  0 ],
                     [0,  0, 0,  1 ]])

    # GLOBAL homogeneous transformation matricesT_w0
    T_w1 = T_w0.dot(T_01)  # from world to link 1
    T_w2 = T_w1.dot(T_12)  # from world to link 2
    T_w3 = T_w2.dot(T_23)  # from world to link 3
    T_w4 = T_w3.dot(T_34)  # from world to link 4
    T_w5 = T_w4.dot(T_45)  # from world to link 5
    T_we = T_w5.dot(T_5e)  # from world to end-effector

    return T_w0, T_w1, T_w2, T_w3, T_w4, T_w5, T_we




def differentKinematics(q):

    # Compute forward kinematics for the different configuration
    T_w0, T_w1, T_w2, T_w3, T_w4, T_w5, T_we = directKinematics(q)

    # link position vectors
    p_w1 = T_w1[:3,3]
    p_w2 = T_w2[:3,3]
    p_w4 = T_w4[:3,3]
    p_w5 = T_w5[:3,3]
    p_we = T_we[:3,3]

    # z vectors for rotations
    z1 = T_w1[:3,2] # Z axis
    z2 = T_w2[:3,2] # Z axis
    z3 = T_w3[:3,2] # Z axis
    z4 = T_w4[:3,2] # Z axis
    z5 = T_w5[:3,2] # Z axis

    # vectors from link i to end-effector
    p_w1e = p_we - p_w1
    p_w2e = p_we - p_w2
    p_w4e = p_we - p_w4
    p_w5e = p_we - p_w5

    # Linear velocity Jacobian (3 × 5)
    J_p = np.hstack((
        np.cross(z1, p_w1e).reshape(3, 1),     # Revolute
        np.cross(z2, p_w2e).reshape(3, 1),     # Revolute
        z3.reshape(3, 1),                      # Prismatic
        np.cross(z4, p_w4e).reshape(3, 1),     # Revolute
        np.cross(z5, p_w5e).reshape(3, 1)      # Revolute
    ))

    # Angular velocity Jacobian (3 × 5)
    J_o = np.hstack((
        z1.reshape(3, 1),                      # Revolute
        z2.reshape(3, 1),                      # Revolute
        np.zeros((3, 1)),                      # Prismatic → zero angular
        z4.reshape(3, 1),                      # Revolute
        z5.reshape(3, 1)                       # Revolute
    ))

    # Combine linear + angular
    J = np.vstack((J_p, J_o))  # 6 × 5

    return J,

    