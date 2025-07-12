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


def rot2rpy(R):
    """
    Convert a rotation matrix to roll-pitch-yaw angles.
    
    :param R: Rotation matrix (3x3).
    :return: Roll, pitch, yaw angles (3-element array).
    """
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = math.atan2(R[1, 0], R[0, 0])
    
    return np.array([roll, pitch, yaw])

def getError(p_desired, pitch_desired, model, data, q):
    """
    Compute the error between the desired end-effector pose and the current pose.
    
    :param p_desired: Desired end-effector position (3-element array).
    :param pitch_desired: Desired end-effector pitch orientation ( around the Y axis )
    :param q0: Initial guess for joint angles (5-element array).
    :return: Error vector (6-element array).
    """
    
    # Compute End Effector pose from current joint angles
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId('mic')
    T_we_current = data.oMf[frame_id].homogeneous  # Get the end-effector pose

    # Position error
    p_current = T_we_current[:3, 3]
    error_position = p_desired - p_current

    # Pitch error
    R_current = T_we_current[:3, :3]
    pitch_current = rot2rpy(R_current)[1]  # Convert rotation matrix to roll-pitch-yaw angles
    pitch_error = pitch_desired - pitch_current
    # Normalize the angular error to the range [-pi, pi]
    # This handles the wrap-around problem and finds the shortest path.
    pitch_error = (pitch_error + np.pi) % (2 * np.pi) - np.pi

    # Combine position and orientation errors
    error = np.hstack((error_position, pitch_error))

    return error, T_we_current

def geometric2analyticJacobian(J,T_0e):
    R_0e = T_0e[:3,:3]              # Extract rotation matrix from the end-effector pose
    rpy_ee = rot2rpy(R_0e)  # Convert rotation matrix to roll-pitch-yaw angles
    roll = rpy_ee[0]                # roll angle
    pitch = rpy_ee[1]               # pitch angle
    yaw = rpy_ee[2]                 # yaw angle    

    # compute the mapping between euler rates and angular velocity
    T_w = np.array([[math.cos(yaw)*math.cos(pitch),  -math.sin(yaw), 0],
                    [math.sin(yaw)*math.cos(pitch),   math.cos(yaw), 0],
                    [             -math.sin(pitch),               0, 1]])

    T_a = np.vstack((
        np.hstack((np.eye(3), np.zeros((3, 3)))),
        np.hstack((np.zeros((3, 3)), np.linalg.inv(T_w)))
    ))

    # Ensure J is 2D
    J_a = T_a @ J  # shape (6,5)
    J_a = np.squeeze(J_a)

    return J_a

def inverseKinematics(p_desired, pitch_desired, model, data, q0=None, max_iter=5, tol=1e-6):
    """
    Perform inverse kinematics to find joint angles that achieve the desired end-effector pose.
    
    :param p_desired: Desired end-effector position (3-element array).
    :param pitch_desired: Desired end-effector pitch orientation ( around the Y axis )
    :param q0: Initial guess for joint angles (5-element array).
    :param max_iter: Maximum number of iterations for convergence.
    :param tol: Tolerance for convergence.
    :return: Joint angles that achieve the desired end-effector pose.
    """
    
    # Initialize variables
    alpha = 1       # Step size
    beta = 0.5      # Step size reduction factor
    damp = 0.01     # Damping factor for numerical stability

    log_grad = []
    log_err = []

    if q0 is None:
        q0 = np.zeros(5)  # Default initial guess
    q = q0.copy()

    
    for iter in range(max_iter):

        # 1 COMPUTE ERROR
        error, T_we_current = getError(p_desired, pitch_desired, model, data, q)  # Compute error

        # 2 COMPUTE NEWTON STEP
        J = differentKinematics(q)        # Compute Jacobian
        # print('Jacobian at iteration {}:\n{}'.format(iter, J))
        # Convert geometric Jacobian to analytic Jacobian to work with pitch error
        J_a = geometric2analyticJacobian(J, T_we_current)
        # print(f"Analitical Jacobian at iteration {iter}:\n{J_a}")
        # Since our task is in 4D (position + pitch), we reduce the Jacobian to the relevant rows
        J_a_reduced = J_a[[0, 1, 2, 4], :]  # row 0,1,2 = position; row 4 = pitch
        # print(f"Jacobian at iteration {iter}:\n{J_a_reduced}")

        # Compute the gradient of the error
        H = np.dot(J_a_reduced, J_a_reduced.T)
        H_reg = H + damp * np.eye(4)
        v = - np.dot(J_a_reduced.T,(np.linalg.inv(H_reg)))  # Gradient descent step
        grad = v.dot(error)
        log_grad.append(grad)

        # 3 DECREASE CRITERION
        count = 0
        while True:
            next_q = q + alpha * grad
            next_error, _= getError(p_desired, pitch_desired, model, data, next_q)
            log_err.append(next_error)
            # Check if the error has decreased
            if np.linalg.norm(error) - np.linalg.norm(next_error) >= 0.0:
                break
            
            print(f"Iteration {iter}: Error did not decrease: {np.linalg.norm(next_error)} >= {np.linalg.norm(error)}")
            print(f"error: {error}, next error: {next_error}")
            print(f"reducing step size from {alpha} to {alpha * beta}")
            alpha = beta * alpha
            count = count +1
            if count > 10:
                print(f"Step size reduction failed after {count} attempts, breaking out of the loop.")
                break

        # 4 CONVERGENCE CHECK
        r = J_a_reduced.T.dot(next_error)
        if np.linalg.norm(r)**2 < tol:
            print(f"Numerical Inverse Kinematics converged in {iter+1} iterations.")
            return next_q  # Convergence achieved

        # If the check fails:
        q = next_q          # Update the current joint angles
        alpha = 1           # Reset step size for the next iteration

    # log printing for debugging
    path = os.path.join(os.path.dirname(__file__), 'ik_debug_log.txt')
    log_file = open(path, "w")  # Open file for writing logs
    for i in range(len(log_grad)):
        log_file.write(f"Iteration {i}: Gradient = {log_grad[i]}, Error = {log_err[i]}\n")
    log_file.close()

    raise ValueError("Inverse kinematics did not converge within the maximum number of iterations.")
