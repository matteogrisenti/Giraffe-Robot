#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

class JointStateForwarder:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.current_positions = [0.0] * len(joint_names)
        self.current_velocities = [0.0] * len(joint_names)
        self.current_efforts = [0.0] * len(joint_names)

        self.pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.sub = rospy.Subscriber('/discrete_joint_states', JointState, self.callback)

        rospy.Timer(rospy.Duration(0.05), self.publish_joint_states)  # 20 Hz

    def callback(self, msg):
        self.current_positions = msg.position
        self.current_velocities = msg.velocity if msg.velocity else [0.0]*len(msg.name)
        self.current_efforts = msg.effort if msg.effort else [0.0]*len(msg.name)

    def publish_joint_states(self, event):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = self.current_positions
        msg.velocity = self.current_velocities
        msg.effort = self.current_efforts
        self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('joint_state_publisher_gui', anonymous=True)
    joint_names = ['shoulder_yaw', 'shoulder_roll', 'prismatic_joint', 'mic_yaw_joint', 'mic_pitch_joint']
    jsp = JointStateForwarder(joint_names)
    rospy.spin()