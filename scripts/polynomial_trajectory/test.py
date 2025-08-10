import os
import json
import rospy
import numpy as np
import pinocchio as pin

from pathlib import Path
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from giraffe_robot.scripts.polynomial_trajectory.polynomial_trajectory import task_domain_polynomial_trajectory

'''
This script is designed to test the polynomial trajectory generation for a robotic arm.
To test the polynomial trajectory the main function:
1. Initializes a ROS node.
2. Loads the URDF model of the robot.
3. Retrieves the initial joint state.
4. Generates a polynomial trajectory based on a desired end-effector 
   position and pitch ( saved in a JSON file).
5. Recovers the polynomial trajectory from the JSON file.
6. Publishes markers to visualize the end-effector trajectory.
7. Publishes the joint states as a trajectory in ROS.
'''


def json_recover_polynomial_trajectory():
    """
    This function recovers the polynomial trajectory from a JSON file.
    It reads the trajectory data and returns the time vector and joint positions.
    """
    path = os.path.join(os.path.dirname(__file__), 'trajectory_data.json')
    with open(path, 'r') as f:
        data = json.load(f)

    timer = np.array(data['t'])
    q_log = np.array(data['q'])
    qd_log = np.array(data['qd'])
    qdd_log = np.array(data['qdd'])

    return timer, q_log, qd_log, qdd_log    


def polynomial_trajectory_publisher(model, data):
    """ 
    This function initializes a ROS node and publishes the polynomial trajectory.
    It reads the trajectory data from a JSON file and publishes it as JointState messages.
    """
    pub = rospy.Publisher('/discrete_joint_states', JointState, queue_size=10)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Recover the polynomial trajectory from the JSON file
    timer, q_log, qd_log, qdd_log = json_recover_polynomial_trajectory()
    # Define the joint names for the robot
    joint_names = ['shoulder_yaw', 'shoulder_roll', 'prismatic_joint', 'mic_yaw_joint', 'mic_pitch_joint']

    total_steps = len(np.array(timer))
    step = 0

    rospy.loginfo("Starting to publish joint trajectory...")

    while not rospy.is_shutdown() and step < total_steps:

        # Recover the numpy structure for the current step
        q_step = np.array(q_log[step])
        qd_step = np.array(qd_log[step])
        qdd_step = np.array(qdd_log[step])

        # Create a JointState message
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        # Fill the message with joint positions, velocities, and efforts
        msg.position = q_step
        msg.velocity = qd_step
        msg.effort = qdd_step

        pub.publish(msg)
        # rospy.loginfo(f"Published positions at t={timer[step]:.2f}: {q_step.tolist()}, {qd_step.tolist()}, {qdd_step.tolist()}")

        step += 1
        rospy.sleep(5)  # Sleep for 5 seconds between each step to simulate the trajectory duration

    rospy.loginfo("Finished publishing trajectory.")


# Initialize the latest joint state variable to be able to read the actual joint state
latest_joint_state = None

def joint_state_callback(msg):
    global latest_joint_state
    latest_joint_state = msg

def get_joint_state():
    """
    This function retrieves the latest joint state from the ROS topic.
    It subscribes to the '/discrete_joint_states' topic and returns the latest JointState message.
    """
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)

    # Wait for a message to be received
    while latest_joint_state is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # print(f"Latest joint state received: {latest_joint_state}")

    return latest_joint_state.position 


if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('polynomial_trajectory_node', anonymous=True)
    rospy.loginfo("Starting polynomial trajectory node...")

    # Define the end effector position and pitch
    position_d = np.array([1, 1, 1])    # Example position
    pitch_d = -0.3                         # Example pitch

    # Load the URDF model
    current_dir = Path(__file__).resolve().parent
    urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    
    # Initialize Pinocchio model and data
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    # Compute the initial joint state
    q0 = np.array(get_joint_state())
    if q0 is None:
        raise RuntimeError("Failed to retrieve the initial joint state.")

    # Generate the polynomial trajectory
    task_domain_polynomial_trajectory(position_d, pitch_d, model, data, q0=q0, duration=5.0)
    timer, q_log, qd_log, qdd_log = json_recover_polynomial_trajectory()
    print("timer:", timer)
    print("timer length:", len(timer))

    # Start the polynomial trajectory publisher
    try:
        polynomial_trajectory_publisher(model, data)
    except rospy.ROSInterruptException:
        rospy.loginfo("Polynomial trajectory publisher interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        rospy.loginfo("Shutting down polynomial trajectory publisher.")


        