#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def create_chair_marker(id, x, y, z=0.5):
    marker = Marker()
    marker.header.frame_id = "floor"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "chairs"
    marker.id = id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.8
    marker.scale.y = 0.8
    marker.scale.z = 0.8

    marker.color.r = 0.2
    marker.color.g = 0.2
    marker.color.b = 0.8
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration(0)
    return marker

def publish_chair_markers():
    rospy.init_node('chair_marker_publisher', anonymous=True)
    pub = rospy.Publisher('chair_markers', Marker, queue_size=10)
    rospy.sleep(1)

    # Chair positions per row
    chair_offsets = [5.4, 4.1, 2.8, 1.5, -1.5, -2.8, -4.1, -5.4]

    # Row X positions
    row_x = [-2, -1, 0, 1, 2]  

    id_counter = 1
    rate = rospy.Rate(1)  # Publish once per second

    while not rospy.is_shutdown():
        for i, x in enumerate(row_x):
            for j, y in enumerate(chair_offsets):
                marker = create_chair_marker(id_counter, x, y)
                pub.publish(marker)
                id_counter += 1
                rospy.sleep(0.05)  # Small delay to ensure visualization
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_chair_markers()
    except rospy.ROSInterruptException:
        pass
