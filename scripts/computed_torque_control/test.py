#!/usr/bin/env python3

"""
test.py

This script tests a **task-space computed torque controller** on a robot model
using Pinocchio for kinematics/dynamics, and ROS for I/O.

Main Features:
- Simulates computed torque control (with or without secondary task in null space)
- Logs debug data for analysis      (directory: **main_secondary_task** if secondary task enable otherwise **main_task**)
- Produces trajectory/error plots   (directory: **main_secondary_task** if secondary task enable otherwise **main_task**)
- Simulate the motion in RVIZ in online or offline mode

Run:
~/giraffe_ws/src$ python3 -m giraffe_robot.scripts.computed_torque_control.test
"""

import os
import json
import time
import asyncio
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import rospy

from sensor_msgs.msg import JointState
from pathlib import Path

# --- Custom modules ---
from .computed_torque_control import ( 
    task_space_torque_control, 
    pd_gains_for_settling_time, 
    task_space_torque_control_with_secondary_task
)
from ..kinematics.kinematics import geometric2analyticJacobian




# =============================================================================================================
#  GOAL TRAJECTORY GENERATION: In order to evaluate the trajectory planned by the controller we need a ideal
#                              trajectory to use like comparison. 
# =============================================================================================================
def generate_desired_task_pos_trajectory(x0, dx0, ddx0, xf, dxf, ddxf, T, dt):
    """
    Generate a smooth **5th-order polynomial trajectory** for each task-space component.

    Parameters
    ----------
    x0, dx0, ddx0 : array-like
        Initial position, velocity, acceleration (n-dim).
    xf, dxf, ddxf : array-like
        Final   position, velocity, acceleration (n-dim).
    T : float
        Total trajectory duration [s].
    dt : float
        Time step [s].

    Returns
    -------
    pos_traj : ndarray, shape (N, n)
        Position trajectory samples.
    t : ndarray, shape (N,)
        Time vector.
    """
    N = int(T / dt) 
    t = np.linspace(0, T, N)
    n = len(x0)
    pos_traj = np.zeros((N, n))

    # Polynomial coefficient matrix (common for all components)
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




# ================================================================
#  TRAJECTORIES PLOTTING UTILITIES
# ================================================================
def controller_plots(x_init, dx_init, ddx_init, x_des, dx_des, ddx_des,
                     x_hist, err_hist, t_hist, T, dt, second_task=True):
    """
    Generate and save plots for:
      - Desired vs. actual task-space trajectory
      - Task-space tracking error

    Saves PNG into:
        - main_task/
        - main_secondary_task/
    depending on whether the secondary task is enabled or not.
    """
    # Convert to numpy arrays for indexing
    x_real_hist = np.array(x_hist)
    err_hist = np.array(err_hist)
    t_hist = np.array(t_hist)

    # Generate target trajectory for comparison
    x_des_hist, _ = generate_desired_task_pos_trajectory(
        x_init, dx_init, ddx_init, x_des, dx_des, ddx_des, T, dt
    )

    # Prepare figure: 2x2 for trajectories + 1 big for error
    fig = plt.figure(figsize=(10, 10))
    labels = ["x", "y", "z", "pitch"]

    # --- Plot trajectories ---
    for i in range(4):
        ax = fig.add_subplot(3, 2, i+1)  # 3 rows, 2 cols, top 4 slots
        ax.plot(t_hist, x_real_hist[:, i], label="Reale")
        ax.plot(t_hist, x_des_hist[:, i], '--', label="Desiderata")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()

    # --- Plot error ---
    ax_err = fig.add_subplot(3, 1, 3)
    if err_hist.ndim > 1:
        for i, lbl in enumerate(labels):
            ax_err.plot(t_hist, err_hist[:, i], label=f"err {lbl}")
    else:
        ax_err.plot(t_hist, err_hist, label="Error norm")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error")
    ax_err.grid(True)
    ax_err.legend()

    plt.tight_layout()

    # Choose output folder
    output_dir = "main_secondary_task" if second_task else "main_task"
    plot_filename = os.path.join(os.path.dirname(__file__), output_dir, "controller_plots.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)




# ==================================================================================================================
#  SIMULATED FEEDBACK LOOP: Since we don't have a real feedback source that return both the task state and velocity, 
#                           we simulate the feedback loop through the direct kinematic.
# ==================================================================================================================
def simulated_feedback_loop(model, data, q, dq):
    # End Effector Current State
    frame_id = model.getFrameId("mic")
    oMf = data.oMf[frame_id]
    T_0e = oMf.homogeneous
    pos = T_0e[:3, 3]           # current position of mic end effector
    R_cur = T_0e[:3, :3]        # current rotation of mic end effector
    
    # Extract Euler angles (roll, pitch, yaw) using pinocchio 
    rpy = pin.rpy.matrixToRpy(R_cur)  # returns (roll, pitch, yaw)
    pitch = float(rpy[1])

    # current spatial velocity in LOCAL_WORLD_ALIGNED
    v_frame = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # v_frame.linear (3,), linear velocity      v 
    # v_frame.angular (3,), angular velocity    w
    
    # End Effector Current State
    frame_id = model.getFrameId("mic")
    oMf = data.oMf[frame_id]
    T_0e = oMf.homogeneous

    # analytic Jacobian (6 x nv) maps qdot -> [x_dot; euler_rates]
    J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ja = geometric2analyticJacobian(J6, T_0e)  # 6 x nv (first 3 linear, next 3 Euler rates in same rpy order)

    # Build reduced task Jacobian Jx (4 x nv): linear(3) + pitch row (the second Euler rate row)
    nv = Ja.shape[1]
    Jx = np.zeros((4, nv))
    Jx[0:3, :] = Ja[0:3, :]              # x,y,z rows   linear part
    Jx[3, :] = Ja[4, :]                  # pitch row    euler part

    # current task state
    x = np.concatenate([pos, [pitch]])
    # map spatial velocity to task velocity: linear + pitch rate
    dx = np.zeros(4)
    dx[0:3] = v_frame.linear
    # Euler rates vector is Ja[3:6,:] @ dq, so compute it explicitly and pick pitch rate
    euler_rates = Ja[3:6, :] @ dq
    dx[3] = euler_rates[1]

    return x, dx
    



# ================================================================
#  ROS COMMUNICATION UTILITIES
# ================================================================
rospy.init_node('Compute_Torque_Control_Test', anonymous=True)
latest_joint_state = None

def joint_state_callback(msg):
    """Callback to store latest joint state from ROS."""
    global latest_joint_state
    latest_joint_state = msg

# ROS Reader
async def get_joint_state():
    """
    Wait for latest joint state from ROS and return as NumPy arrays.
    From the joint configuration through direct kinematics we derive the end-effector pose and velocity
    and use them to derive the error. 
    """
    global latest_joint_state
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)

    while latest_joint_state is None and not rospy.is_shutdown():
        rospy.sleep(0.01)

    # Convert to NumPy arrays for Pinocchio
    position = np.array(latest_joint_state.position, dtype=np.float64)
    velocity = np.array(latest_joint_state.velocity, dtype=np.float64)
    latest_joint_state = None
    return position, velocity

async def get_task_state():
    """
    This function should read the current task state (position and velocity) from a specific ROS topic
    and return it as a tuple (x, dx). We have a ROS topic for task state but, unfortunately, we don't have a 
    direct ROS topic for task velocity, so we will use the simulated feedback loop instead.
    """
    

# ROS Publisher
pub = rospy.Publisher('/discrete_joint_states', JointState, queue_size=10)
# NB: we publish on /discrete_joint_states; the values are then read by the node joint_state_publisher_gui
#     which publish them on /joint_states

async def set_joint_state(q, dq, ddq):
    """Publish joint state (pos, vel, effort) to ROS."""
    joint_names = [
        'shoulder_yaw', 'shoulder_roll', 'prismatic_joint',
        'mic_yaw_joint', 'mic_pitch_joint'
    ]
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = joint_names
    msg.position = q
    msg.velocity = dq
    msg.effort = ddq
    pub.publish(msg)


# NB: The function save_joint_state is used to save the joint state in a json file, this allow to store 
#     the trajectory and execute it in a second moment

async def save_joint_state(q, dq, ddq, t, joint_state_log):
    """Save joint state to json structure"""
    joint_names = [
        'shoulder_yaw', 'shoulder_roll', 'prismatic_joint',
        'mic_yaw_joint', 'mic_pitch_joint'
    ]   

    joint_state_log.append({
    "timestamp": t,
    "joint_states": {
        "name": joint_names,
        "position": q.tolist(),
        "velocity": dq.tolist(),
        "effort": ddq.tolist()
    }
})
    
def save_json_joint_state(joint_state_log, output_dir):
    """Save joint state log to a JSON file."""
    log_path = os.path.join(os.path.dirname(__file__), output_dir, "joint_state_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        json.dump(joint_state_log, f, indent=4)


# ================================================================
#  SIMULATION CORE
# ================================================================
async def simulation_computed_torque_control(
    model, q0, dq0, frame_id,
    x_des, dx_des, ddx_des,
    Kp=None, Kd=None, dt=1e-3, steps=100,
    debug=False, second_task=True, log_filename="log.txt"
):
    """
    Simulate **inverse dynamics control in task space** with a feedback loop.

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
    second_task : bool
        If True, use the compute torque control law that achieves the secondary task in the null space.
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

    # Logging containers
    x_hist, err_hist, t_hist = [], [], []
    joint_state_log = []

     # Prepare debug log file if requested
    if debug:
        # Open log file (overwrite mode)

        # organize output directory to maintain order directory, divide the output in two directory:
        # - main_task: for the experiments that don't exploit the null space of the 
        # - main_secondary_task: for the experiments that do exploit the null space of the 
        output_dir = "main_task"
        if second_task:
            output_dir = "main_secondary_task"    

        path = os.path.join(os.path.dirname(__file__), output_dir, log_filename)
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
        # --- Compute control torques ---
        try:
            # NB: The read of the joint state has been deactivated, this allow more fast execution crucial for the development phase
            # this don't introduce error becouse we would reed the value published by the privius iteration that corrispond to q, dq
            # q, dq = await get_joint_state()         # read joint state and velocity from ros

            # --- Update the robot state ---
            pin.forwardKinematics(model, data, q, dq)
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            
            # NB:In addition the value that we would read would be the same that we compute in the previous step, so in this configuration of the simulation we don't use a real
            # feedback loop but only a simulated one.
            # x, dx, = await get_task_state()        # read task state and velocity from ros
            x, dx = simulated_feedback_loop(model, data, q, dq)  # simulate the feedback loop

            # Decide which control law to use
            if second_task:
                tau = task_space_torque_control_with_secondary_task(
                    model, data, q, dq, x, dx, x_des, dx_des, ddx_des, Kp=Kp, Kd=Kd, q0=q0, Kq=None, Dq=None
                )
            else:
                tau = task_space_torque_control(
                    model, data, q, dq, x, dx, x_des, dx_des, ddx_des, Kp=Kp, Kd=Kd
                )

        except ValueError as e:
            if debug:
                log_file.write(f"Error at step {i}: {e}\n")
                log_file.close()
            print(f"Error at step {i}: {e}")
            return q, dq, np.zeros_like(dq)


        # --- Simulate robot dynamics ---
        # Compute dynamics: M * ddq + h = tau → ddq = M⁻¹ (tau - h)
        M = pin.crba(model, data, q)                         # Mass matrix
        h = pin.rnea(model, data, q, dq, np.zeros_like(dq))  # Bias forces (Coriolis, gravity)
        ddq = np.linalg.solve(M, tau - h)                    # Joint accelerations

        # Integrate to update velocities and positions (Euler integration)
        dq += ddq * dt
        q = pin.integrate(model, q, dq * dt)  # Integrates over manifold (better than q += dq * dt)

        # NB: The publication of the joint state has been deactivated, this allow more fast execution crucial for the development phase
        # For the limitation of my hardware i decide to store the trajectory in a json file and execute it in a second moment
        # await set_joint_state(q, dq, tau)
        await save_joint_state(q, dq, ddq, i * dt, joint_state_log)  # Save joint state to JSON file

        # check for NaN or inf in joint positions and velocities
        if np.any(np.isnan(q)) or np.any(np.isinf(q)) or np.any(np.isnan(dq)) or np.any(np.isinf(dq)):
            print(f"Warning: NaN or inf detected in joint states at step {i}!")
            print(f"q: {q}, dq: {dq}, tau: {tau}")
            if debug:
                log_file.write(f"Warning: NaN or inf detected in joint states at step {i}!\n")
                log_file.write(f"q: {q}, dq: {dq}, tau: {tau}\n")
            raise ValueError("NaN or inf detected in joint states!")  


        # --- Compute the new task pose ---
        pin.forwardKinematics(model, data, q, dq)
        oMf = data.oMf[frame_id]
        pos = oMf.translation
        R = oMf.rotation
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        x = np.concatenate([pos, [pitch]])

        # Compute the error
        err = x_des - x

        # Store history
        x_hist.append(x)
        err_hist.append(err)
        t_hist.append(i * dt)

        # Debug logging evry ten steps
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


    save_json_joint_state(joint_state_log, output_dir=output_dir)  # Save joint state log to JSON file
    return q, dq, ddq, x_hist, err_hist, t_hist




# ================================================================
#  MAIN TEST FUNCTION
# ================================================================
async def test_task_space_inverse_dynamics(second_task=True):
    """
    Load a robot URDF and test the task-space inverse dynamics controller.

    Par:
    - second_task: Whether to use the secondary task in null space for control.
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

    # Simulation params
    dt = 1e-3           # Time of each step
    T = 7.0             # Total time in seconds
    steps = int(T / dt)

    # Initial pose
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

    # Desired state (position + pitch)
    x_des = np.array([1, 2, 1, 0.3])                
    dx_des = np.array([0.0, 0.0, 0.0, 0.0])         # Target velocity
    ddx_des = np.array([0.00, 0.00, 0.00, 0.00])    # Target acceleration

    print(f"Desired task-space position + pitch: {x_des}")
    print(f"Desired task-space velocity: {dx_des}")
    print(f"Desired task-space acceleration: {ddx_des}")

    Kp_rec, Kd_rec = pd_gains_for_settling_time(Ts=T)

    print("Recommended Kp diag:", np.diag(Kp_rec))
    print("Recommended Kd diag:", np.diag(Kd_rec))

    # Run simulation
    print("\nRunning simulation...")
    q, dq, ddq, x_hist, err_hist, t_hist = await simulation_computed_torque_control(
        model, q, dq, frame_id, x_des, dx_des, ddx_des,
        Kp=Kp_rec, Kd=Kd_rec, dt=1e-3, steps=steps, second_task=second_task,debug=True
    )

    # Plot results
    controller_plots(
        x_init, dx_init, ddx_init, x_des, dx_des, ddx_des, x_hist, 
        err_hist, t_hist, T, dt, second_task=second_task
    )


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




# ================================================================
#  OFLINE RVIZ SIMULATION
# ================================================================
def rviz_simulation(second_task=True, rate_hz=1000):
    output_dir = "main_task"
    if second_task:
        output_dir = "main_secondary_task"  

    json_file = os.path.join(os.path.dirname(__file__), output_dir, "joint_state_log.json")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    # Load trajectory
    with open(json_file, 'r') as f:
        trajectory = json.load(f)

    if not isinstance(trajectory, list):
        raise ValueError("JSON trajectory must be a list of joint states.")

    rate = rospy.Rate(rate_hz)
    print(f"\nReplaying {len(trajectory)} states at {rate_hz} Hz...")
    start_time = time.time()

    for entry in trajectory:
        if rospy.is_shutdown():
            break

        msg = JointState()
        msg.header.stamp = rospy.Time.now()

        # Extract data
        joint_data = entry["joint_states"]
        msg.name = joint_data["name"]
        msg.position = joint_data["position"]
        msg.velocity = joint_data["velocity"]
        msg.effort = joint_data["effort"]

        pub.publish(msg)
        rate.sleep()

    print("Trajectory replay complete.")




# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    second_task = False  # Set to False to disable secondary task

    # Run the Controller to generate the trajectory
    # asyncio.run(test_task_space_inverse_dynamics(second_task=second_task))

    # Run the Simulation
    rviz_simulation(second_task=second_task, rate_hz=1000)














