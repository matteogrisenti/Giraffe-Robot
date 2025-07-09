# This file contains configuration parameters for the Giraffe robot's URDF model.
# It defines various properties such as dimensions and lengths of different parts of the robot.

# configuration parameters for the Giraffe robot URDF model
shoulder_joint_radius = 0.05  # shoulder joint radius
shoulder_joint_height = 0.1  # shoulder joint height
arm_length = 2.4  # arm length
arm_radius = 0.05  # arm radius
prismatic_height_delta = 0.1  # prismatic extension height delta
wrist_joint_height = 0.1  # wrist joint height
wrist_radius = 0.025  # wrist joint radius
wrist_link_length = 1.0  # wrist link length
mic_stick_length = 0.15  # microphone stick length
mic_stick_radius = 0.015  # microphone stick radius


a1 = shoulder_joint_height
a3_x = shoulder_joint_radius + arm_length + wrist_radius
a3_y = prismatic_height_delta
a5 = wrist_link_length
a6 = mic_stick_length + 2*mic_stick_radius

