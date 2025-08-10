#!/usr/bin/env python3

import os
import asyncio
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import rospy

from sensor_msgs.msg import JointState
from pathlib import Path

from .computed_torque_control import task_space_torque_control, pd_gains_for_settling_time
from ..kinematics.kinematics import geometric2analyticJacobian


# FUNCTION TO PLTO THE RESULT OF THE TEST
def generate_desired_task_pos_trajectory(x0, dx0, ddx0, xf, dxf, ddxf, T, dt):
    """
    Genera una traiettoria 5° ordine per ogni componente di x.
    Input: x0, dx0, ddx0, xf, dxf, ddxf -> vettori (dim = n)
    Output: pos_traj (N x n), tempo (N,)
    """
    N = int(T / dt) 
    t = np.linspace(0, T, N)
    n = len(x0)
    pos_traj = np.zeros((N, n))

    # Matrice comune a tutte le componenti
    M = np.array([
        [0,     0,    0,   0,   0, 1],
        [T**5, T**4, T**3, T**2, T, 1],
        [0,     0,    0,   0,   1, 0],
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [0,     0,    0,   2,   0, 0],
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]
    ])

    for k in range(n):
        b = np.array([x0[k], xf[k], dx0[k], dxf[k], ddx0[k], ddxf[k]])
        a = np.linalg.solve(M, b)
        pos_traj[:, k] = np.polyval(a, t)

    return pos_traj, t



def controller_plots(x_init, dx_init, ddx_init, x_des, dx_des, ddx_des,
                     x_hist, err_hist, t_hist, T, dt):
    '''
    Funzione per generare i grafici di controllo
    '''
    # Convert arrays to numpy
    x_real_hist = np.array(x_hist)
    err_hist = np.array(err_hist)
    t_hist = np.array(t_hist)

    # Compute the target pose trajectory
    x_des_hist, _ = generate_desired_task_pos_trajectory(
        x_init, dx_init, ddx_init, x_des, dx_des, ddx_des, T, dt
    )

    # Prepare figure: 2x2 for trajectories + 1 big for error
    fig = plt.figure(figsize=(10, 10))
    labels = ["x", "y", "z", "pitch"]

    # 2x2 grid for trajectories
    for i in range(4):
        ax = fig.add_subplot(3, 2, i+1)  # 3 rows, 2 cols, top 4 slots
        ax.plot(t_hist, x_real_hist[:, i], label="Reale")
        ax.plot(t_hist, x_des_hist[:, i], '--', label="Desiderata")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()

    # Error plot: last row spans 2 columns
    ax_err = fig.add_subplot(3, 1, 3)  # 3 rows, 1 col, last row
    if err_hist.ndim > 1 and err_hist.shape[1] > 1:
        # Plot all component errors
        for i, lbl in enumerate(labels):
            ax_err.plot(t_hist, err_hist[:, i], label=f"err {lbl}")
        ax_err.legend()
    else:
        # Plot scalar error norm
        ax_err.plot(t_hist, err_hist, label="Error norm")
        ax_err.legend()

    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error")
    ax_err.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(os.path.dirname(__file__), "controller_plots.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)




# ROS FUNCTIONS
rospy.init_node('Compute_Torque_Control_Test', anonymous=True)

# ROS READING SYSTEM
# Initialize the latest joint state variable to be able to read the actual joint state
latest_joint_state = None

def joint_state_callback(msg):
    global latest_joint_state
    latest_joint_state = msg

async def get_joint_state():
    # print("Waiting for joint states...")
    global latest_joint_state

    rospy.Subscriber('/joint_states', JointState, joint_state_callback)

    while latest_joint_state is None and not rospy.is_shutdown():
        rospy.sleep(0.01)  # Wait for the joint state to be received

    position = latest_joint_state.position
    velocity = latest_joint_state.velocity

    # Convoert it in numpy so pinocchio can use them
    position = np.array(position, dtype=np.float64)
    velocity = np.array(velocity, dtype=np.float64)

    latest_joint_state = None
    return position, velocity

# ROS WRITING SYSTEM
pub = rospy.Publisher('/discrete_joint_states', JointState, queue_size=10)

async def set_joint_state(q, dq, ddq):
    joint_names = [
        'shoulder_yaw', 
        'shoulder_roll', 
        'prismatic_joint', 
        'mic_yaw_joint', 
        'mic_pitch_joint'
    ]

    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = joint_names
    msg.position = q
    msg.velocity = dq
    msg.effort = ddq

    pub.publish(msg)




async def simulation_computed_torque_control(model, q0, dq0, frame_id, x_des, dx_des, ddx_des, Kp=None, Kd=None, dt=1e-3, steps=100, debug=False, log_filename="log.txt"):
    """
    Simulates inverse dynamics control in task space with a simulated feddback loop

    Parameters
    ----------
    model : pinocchio.Model
        The Pinocchio model of the robot.
    q0 : ndarray
        Initial joint positions.
    dq0 : ndarray
        Initial joint velocities.
    frame_id : int
        ID of the frame used as task-space control target (e.g., end-effector).
    x_des : ndarray
        Desired task-space position and pitch [x, y, z, pitch].
    dx_des : ndarray
        Desired task-space velocity.
    ddx_des : ndarray
        Desired task-space acceleration.
    Kp : ndarray       
        Task-space proportional gain (4x4)
    Kd : ndarray        
        Task-space derivative gain (4x4)
    dt : float
        Time step (seconds).
    steps : int
        Number of simulation steps.
    debug : bool
        If True, enables debug output.
    log_filename : text
        Path to the log file for saving debug information.

    Returns
    -------
    q : ndarray
        Final joint positions.
    dq : ndarray
        Final joint velocities.
    ddq : ndarray
        Final joint accelerations.
    """

    data = model.createData()
    q = q0.copy()
    dq = dq0.copy()

    # to store the real trajectory in order to plot it at the end
    x_hist = []
    err_hist = []
    t_hist = []


    if debug:
        # Open log file (overwrite mode)
        path = os.path.join(os.path.dirname(__file__), log_filename)
        log_file = open(path, "w", encoding="utf-8")

        log_file.write("=== Computed Torque Control Simulation Log ===\n")
        log_file.write(f"Initial q: {q0}\n")
        log_file.write(f"Initial dq: {dq0}\n")
        log_file.write(f"Desired x: {x_des}\n")
        log_file.write(f"Desired dx: {dx_des}\n")
        log_file.write(f"Desired ddx: {ddx_des}\n")
        log_file.write(f"Kp: {Kp}\n")
        log_file.write(f"Kd: {Kd}\n")
        log_file.write(f"dt: {dt}, steps: {steps}\n")
        log_file.write("="*60 + "\n")
    
    for i in range(steps):
        # Compute joint torques using task-space torque control
        try:
            # q, dq = await get_joint_state()         # read joint state and velocity from ros
            tau = task_space_torque_control(model, data, q, dq, x_des, dx_des, ddx_des, Kp=Kp, Kd=Kd)
        except ValueError as e:
            if debug:
                log_file.write(f"Error at step {i}: {e}\n")
                log_file.close()
            print(f"Error at step {i}: {e}")
            return q, dq, np.zeros_like(dq)


        # Simulate dynamics
        # Compute dynamics: M * ddq + h = tau → ddq = M⁻¹ (tau - h)
        M = pin.crba(model, data, q)  # Mass matrix
        h = pin.rnea(model, data, q, dq, np.zeros_like(dq))  # Bias forces (Coriolis, gravity)
        ddq = np.linalg.solve(M, tau - h)  # Joint accelerations

        # Integrate to update velocities and positions (Euler integration)
        dq += ddq * dt
        q = pin.integrate(model, q, dq * dt)  # Integrates over manifold (better than q += dq * dt)

        # await set_joint_state(q, dq, tau)

        # check for NaN or inf in joint positions and velocities
        if np.any(np.isnan(q)) or np.any(np.isinf(q)) or np.any(np.isnan(dq)) or np.any(np.isinf(dq)):
            print(f"Warning: NaN or inf detected in joint states at step {i}!")
            print(f"q: {q}, dq: {dq}, tau: {tau}")
            if debug:
                log_file.write(f"Warning: NaN or inf detected in joint states at step {i}!\n")
                log_file.write(f"q: {q}, dq: {dq}, tau: {tau}\n")
            raise ValueError("NaN or inf detected in joint states!")    
        
        # Compute the task pose at the end of the step
        pin.forwardKinematics(model, data, q, dq)
        oMf = data.oMf[frame_id]
        pos = oMf.translation
        R = oMf.rotation
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        x = np.concatenate([pos, [pitch]])

        # Compute the error
        err = x_des - x

        # Salva storico
        x_hist.append(x)
        err_hist.append(err)
        t_hist.append(i * dt)

        # Debug logging every 10 steps or at last step
        if debug and (i % 10 == 0 or i == steps - 1):
        
            # Approximate task-space velocity
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dx = J[:4, :] @ dq  # only first 4 rows if relevant

            log_file.write(f"[step {i}]\n")
            log_file.write(f"  Task-space pos+pitch: {x}\n")
            log_file.write(f"  Task-space vel (approx): {dx}\n")
            log_file.write(f"  Error pos: {err[:3]}, pitch err: {err[3]}\n")
            log_file.write(f"  Desired accel: {ddx_des}\n")
            log_file.write(f"  Tau: {tau}, ||tau||={np.linalg.norm(tau):.4f}\n")
            log_file.write(f"  Joint positions: {q}\n")
            log_file.write(f"  Joint velocities: {dq}\n")
            log_file.write(f"  Joint accelerations: {ddq}\n")
            log_file.write("-"*40 + "\n")

    if debug:
        log_file.write("=== Simulation Complete ===\n")
        log_file.write(f"Final q: {q}\n")
        log_file.write(f"Final dq: {dq}\n")
        log_file.write(f"Final ddq: {ddq}\n")
        log_file.close()


    return q, dq, ddq, x_hist, err_hist, t_hist




async def test_task_space_inverse_dynamics():
    """
    Load a robot URDF and test the task-space inverse dynamics controller.
    """

    # Define the URDF file path
    current_dir = Path(__file__).resolve().parent
    urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    
    # Build the robot model from the URDF
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    
    # Frame used for task-space control (e.g., end-effector)
    frame_id = model.getFrameId("mic")

    # read joint configuration (home position)
    q = np.zeros(model.nq)
    dq = np.zeros(model.nv)
    q, dq = await get_joint_state()

    # Initialize the time
    dt = 1e-3           # Time of each step
    T = 7.0             # Total time in seconds
    steps = int(T / dt)

    # Compute current end-effector pose
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[frame_id]
    pos = oMf.translation
    R = oMf.rotation
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))

    x_init = np.concatenate([pos, [pitch]])
    dx_init = np.zeros(4)       # Initial task-space velocity
    ddx_init = np.zeros(4)      # Initial task-space acceleration

    print(f"Initial task-space position + pitch: {x_init}")

    # Define desired task-space target (position + pitch)
    x_des = np.array([1, 1, 1, 0.3])       # This ensures your test uses the locally linear assumption (required for computed torque control).
    dx_des = np.array([0.0, 0.0, 0.0, 0.0])         # Target velocity
    ddx_des = np.array([0.00, 0.00, 0.00, 0.00])    # Target acceleration

    print(f"Desired task-space position + pitch: {x_des}")
    print(f"Desired task-space velocity: {dx_des}")
    print(f"Desired task-space acceleration: {ddx_des}")

    Kp_rec, Kd_rec = pd_gains_for_settling_time(Ts=T)

    print("Recommended Kp diag:", np.diag(Kp_rec))
    print("Recommended Kd diag:", np.diag(Kd_rec))

    # Run the simulation
    print("\nRunning simulation...")
    q, dq, ddq, x_hist, err_hist, t_hist = await simulation_computed_torque_control(
        model, q, dq, frame_id, x_des, dx_des, ddx_des,
        Kp=Kp_rec, Kd=Kd_rec, dt=1e-3, steps=steps, debug=True
    )

    # Plot results
    controller_plots(x_init, dx_init, ddx_init, x_des, dx_des, ddx_des, x_hist, err_hist, t_hist, T, dt)


    # Final task-space state
    pin.forwardKinematics(model, data, q, dq)       # Compute forward kinematics
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[frame_id]
    pos_final = oMf.translation                     # get current position
    R_final = oMf.rotation                          # get current orientation
    pitch_final = np.arctan2(-R_final[2, 0], np.sqrt(R_final[0, 0]**2 + R_final[1, 0]**2))
    x_final = np.concatenate([pos_final, [pitch_final]])

    print(f"\nFinal task-space position + pitch: {x_final}")


    # --- Final task-space velocity ---
    J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ja = geometric2analyticJacobian(J6, oMf.homogeneous)
    # Reduce to [x, y, z, pitch]
    dx_final = np.zeros(4)
    dx_final[0:3] = Ja[0:3, :] @ dq
    euler_rates_final = Ja[3:6, :] @ dq
    dx_final[3] = euler_rates_final[1]  # pitch rate

    print(f"Final task-space velocity: {dx_final}")

    # --- Final task-space acceleration ---
    ddx_final = np.zeros(4)
    ddx_final[0:3] = Ja[0:3, :] @ ddq
    euler_acc_final = Ja[3:6, :] @ ddq
    ddx_final[3] = euler_acc_final[1]  # pitch angular acceleration

    print(f"Final task-space acceleration: {ddx_final}")



# Run the test when script is executed directly
asyncio.run(test_task_space_inverse_dynamics())














