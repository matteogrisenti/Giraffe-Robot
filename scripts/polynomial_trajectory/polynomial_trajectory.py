import os
import json
import pinocchio as pin
import numpy as np

from scripts.kinematics.kinematics import inverseKinematics


def fifthOrderPolynomialCoefficients(tf, q0 , qf , qd0 = np.zeros(5), qdf = np.zeros(5), qdd0 = np.zeros(5), qddf = np.zeros(5)):
    """
    Computes the coefficients of a fifth-order polynomial trajectory.
    The polynomial is defined as:
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    where the coefficients are determined based on the start and end positions, velocities, and accelerations.
    Args:
        tf (float): Final time of the trajectory.
        q0 (np.ndarray): Start configuration.
        qf (np.ndarray): End configuration.
        qd0 (np.ndarray, optional): Start velocity. Defaults to 0.
        qdf (np.ndarray, optional): End velocity. Defaults to 0.
        qdd0 (np.ndarray, optional): Start acceleration. Defaults to 0.
        qddf (np.ndarray, optional): End acceleration. Defaults to 0.
    Returns:
        np.ndarray: Coefficients of the polynomial trajectory.
    """

    # Matrix used to solve the linear system of equations for the polynomial trajectory
    polyMatrix = np.array([[1,  0,              0,               0,                  0,                0],
                           [0,  1,              0,               0,                  0,                0],
                           [0,  0,              2,               0,                  0,                0],
                           [1, tf,np.power(tf, 2), np.power(tf, 3),    np.power(tf, 4),  np.power(tf, 5)],
                           [0,  1,           2*tf,3*np.power(tf,2),   4*np.power(tf,3), 5*np.power(tf,4)],
                           [0,  0,              2,             6*tf, 12*np.power(tf,2),20*np.power(tf,3)]])
    
    # Vector containing the start and end positions, velocities, and accelerations
    polyVector = np.array([q0, qd0, qdd0, qf, qdf, qddf])
    matrix_inv = np.linalg.inv(polyMatrix)
    polyCoeff = matrix_inv.dot(polyVector)

    return polyCoeff




def polynomial_trajectory(q0, qf, qd0 = np.zeros(5), qdf = np.zeros(5), qdd0 = np.zeros(5), qddf = np.zeros(5), duration=5.0):
    """
    This function generates a polynomial trajectory for a robot arm.
    It computes the coefficients for a polynomial that describes the motion of the end-effector.
    """
    
    # Compute polynomial coefficients
    coeffs = fifthOrderPolynomialCoefficients(duration, q0, qf, qd0, qdf, qdd0, qddf)

    # Create a time vector for the trajectory
    timer = np.linspace(0, duration, num=10)

    # Initialize lists to store joint positions, velocities, and accelerations
    t_log = []
    q_log = []
    qd_log = []
    qdd_log = []

    for t in timer:
        # Compute the polynomial trajectory at time t
        q = coeffs[0] + coeffs[1] * t + coeffs[2] * t**2 + coeffs[3] * t**3 + coeffs[4] * t**4 + coeffs[5] * t**5
        qd = coeffs[1] + 2 * coeffs[2] * t + 3 * coeffs[3] * t**2 + 4 * coeffs[4] * t**3 + 5 * coeffs[5] * t**4
        qdd = 2 * coeffs[2] + 6 * coeffs[3] * t + 12 * coeffs[4] * t**2 + 20 * coeffs[5] * t**3
        
        # Append the computed values to the logs
        t_log.append(t)
        q_log.append(q.tolist())  # Convert to list for JSON serialization
        qd_log.append(qd.tolist())
        qdd_log.append(qdd.tolist())

    # save the trajectory data in a json file
    trajectory_data = {
        "t": t_log,
        "q": q_log,
        "qd": qd_log,
        "qdd": qdd_log,
    }

    # print("Polynomial trajectory data:")
    # print("Time (t):", t_log)
    # print("Joint positions (q):", q_log)
    # print("Joint velocities (qd):", qd_log)
    # print("Joint accelerations (qdd):", qdd_log)

    # Save the trajectory data to a JSON file
    path = os.path.join(os.path.dirname(__file__), 'trajectory_data.json')
    with open(path, 'w') as f:
        json.dump(trajectory_data, f, indent=4)

    print("Polynomial Trajectory Generated, data saved to 'trajectory_data.json'.")




def task_domain_polynomial_trajectory(postion_d, pitch_d, model, data, q0 = np.zeros(5), duration=5.0):
    """
    This function is a task domain polynomial trajectory generator. 
    It takes a desired position and pitch, compute the q_d with inverse kinematics
    and computes the polynomial trajectory coefficients.
    """

    print("\nDesired task configuration: pose", postion_d, " pitch", pitch_d)

    # Compute inverse kinematics to get the desired joint positions
    q_d = inverseKinematics(postion_d, pitch_d, model, data, q0=q0)
    print("Desired joint configuration (IK):", q_d)

    # Generate the polynomial trajectory, it is saved in a json file
    polynomial_trajectory( q0=q0, qf=q_d, duration=duration )




  