import os
import sys
import json
import rospy
import numpy as np
import pinocchio as pin

from pathlib import Path
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from scripts.polynomial_trajectory.polynomial_trajectory import task_domain_polynomial_trajectory
from scripts.kinematics.kinematics import rot2rpy

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

Run:
~/giraffe_ws/src/giraffe_robot$ python3 -m scripts.polynomial_trajectory.test
'''



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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



# -----------------------------------------------------------------------------
# ROS COMUNICATION UTILITIES
# -----------------------------------------------------------------------------
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




def polynomial_trajectory_publisher(model, data):
    """ 
    This function initializes a ROS node and publishes the polynomial trajectory.
    It reads the trajectory data from a JSON file and publishes it as JointState messages.
    When it end each step it wait for a short duration to make more clear the steps distinctions
    """
    pub = rospy.Publisher('/discrete_joint_states', JointState, queue_size=10)
    pose_pub = rospy.Publisher('/polynomial_trajectory_visualization_marker', PoseArray, queue_size=20)
    rospy.sleep(1.0)
    rate = rospy.Rate(10)  # 10 Hz

    # Recover the polynomial trajectory from the JSON file
    timer, q_log, qd_log, qdd_log = json_recover_polynomial_trajectory()
    # Define the joint names for the robot
    joint_names = ['shoulder_yaw', 'shoulder_roll', 'prismatic_joint', 'mic_yaw_joint', 'mic_pitch_joint']

    total_steps = len(np.array(timer))
    step = 0

    # Publish the poses
    pose_array = PoseArray()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = "base_link" 

    for q in q_log:
        frame_id = model.getFrameId('mic')
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        step_pose = data.oMf[frame_id].homogeneous

        step_position= step_pose[:3, 3]
        
        pose_msg = Pose()
        pose_msg.position.x = float(step_position[0])
        pose_msg.position.y = float(step_position[1])
        pose_msg.position.z = float(step_position[2])
        pose_msg.orientation.w = 1.0  # identità, nessuna rotazione
        pose_array.poses.append(pose_msg)

    # Publish the poses
    # print("Publishing poses:", [pose.position for pose in pose_array.poses])
    pose_pub.publish(pose_array)

    print('\n')
    rospy.loginfo("Starting to publish joint trajectory on ROS")

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
        rospy.sleep(20)  # Sleep for 5 seconds between each step to simulate the trajectory duration

    rospy.loginfo("Finished publishing trajectory.")




if __name__ == '__main__':
    # Check if ROS master node is running
    master = rospy.get_master()
    try:
        master.getSystemState()  # fa una vera richiesta XML-RPC
    except Exception:
        sys.stderr.write(
            "\nROS master node is not running, start it before run this test.\n"
            "1) ~/giraffe_ws$ source devel/setup.bash\n"
            "2) ~/giraffe_ws$ roslaunch giraffe_robot simulation.launch\n\n"
        )
        sys.exit(1)

    # Initialize the ROS node
    rospy.init_node('polynomial_trajectory_node', anonymous=True)
    rospy.loginfo("Started polynomial trajectory node")

    # Define the end effector position and pitch
    position_d = np.array([1, 2, 1])       # Example position
    pitch_d = -0.3                         # Example pitch

    # Load the URDF model
    current_dir = Path(__file__).resolve().parent
    urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    
    # Initialize Pinocchio model and data
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    # Read the initial joint state
    q0 = np.array(get_joint_state())
    if q0 is None:
        raise RuntimeError("Failed to retrieve the initial joint state.")

    # Generate the polynomial trajectory offline
    task_domain_polynomial_trajectory(position_d, pitch_d, model, data, q0=q0, duration=5.0)

    # Recover the polynomial trajectory
    timer, q_log, qd_log, qdd_log = json_recover_polynomial_trajectory()
    print("\nnumber of steps :", len(timer))
    print("steps time:", timer)

    # Start the polynomial trajectory publisher
    try:
        polynomial_trajectory_publisher(model, data)
    except rospy.ROSInterruptException:
        rospy.loginfo("Polynomial trajectory publisher interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        rospy.loginfo("Shutting down polynomial trajectory publisher.")

    # Check the reached pose
    current_q = np.array(get_joint_state())     # read current joint state
    if current_q is None: raise RuntimeError("Failed to retrieve the current joint state.")

    frame_id = model.getFrameId('mic')
    pin.forwardKinematics(model, data, current_q)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[frame_id].homogeneous

    current_position = current_pose[:3, 3]
    current_orientation = current_pose[:3, :3]
    current_pitch = rot2rpy(current_orientation)[1]

    print('\nReached Position: ', current_position)
    print('Reached Pitch: ', current_pitch)

    print('\nError Position: ', current_position - position_d)
    print('Error Pitch: ', current_pitch - pitch_d)
