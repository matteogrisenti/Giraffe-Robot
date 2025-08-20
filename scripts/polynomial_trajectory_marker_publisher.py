#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, Point

# Lista globale per salvare i punti della traiettoria
trajectory_points = []

def create_step_marker(x, y, z, marker_id):    
    sphere_marker = Marker()
    sphere_marker.header.frame_id = "world"
    sphere_marker.header.stamp = rospy.Time.now()
    sphere_marker.ns = "trajectory_points"
    sphere_marker.id = marker_id
    sphere_marker.type = Marker.SPHERE
    sphere_marker.action = Marker.ADD
    sphere_marker.pose.position.x = x
    sphere_marker.pose.position.y = y
    sphere_marker.pose.position.z = z
    sphere_marker.pose.orientation.w = 1.0
    sphere_marker.scale.x = 0.05
    sphere_marker.scale.y = 0.05
    sphere_marker.scale.z = 0.05
    sphere_marker.color.r = 1.0
    sphere_marker.color.g = 0.0
    sphere_marker.color.b = 0.0
    sphere_marker.color.a = 1.0
    sphere_marker.lifetime = rospy.Duration()
    return sphere_marker

def create_trajectory_line_marker(points):  
    line_marker = Marker()
    line_marker.header.frame_id = "world"
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = "trajectory_line"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.01
    line_marker.color.r = 0.0
    line_marker.color.g = 0.0
    line_marker.color.b = 1.0
    line_marker.color.a = 1.0
    line_marker.points = points
    return line_marker

def visualize_trajectory(msg, marker_pub):
    global trajectory_points
    markers = []

    # Aggiungi i punti ricevuti al globale
    # rospy.loginfo("Received PoseArray with %d poses", len(msg.poses))
    for i, pose in enumerate(msg.poses):
        pt = Point()
        pt.x = pose.position.x
        pt.y = pose.position.y
        pt.z = pose.position.z
        trajectory_points.append(pt)
        markers.append(create_step_marker(pt.x, pt.y, pt.z, i + 100))
    
    markers.append(create_trajectory_line_marker(trajectory_points))

    for marker in markers:
        marker_pub.publish(marker)

def trajectory_marker_publisher():
    rospy.init_node('polynomial_trajectory_marker_publisher')
    marker_pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=20)
    rospy.sleep(1.0)

    rospy.Subscriber('/polynomial_trajectory_visualization_marker', PoseArray, visualize_trajectory, marker_pub)
    rospy.loginfo("Nodo trajectory_marker_publisher avviato. In attesa di PoseArray...")
    rospy.spin()

if __name__ == '__main__':
    try:
        trajectory_marker_publisher()
    except rospy.ROSInterruptException:
        pass