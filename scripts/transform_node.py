#!/usr/bin/env python

import rospy
import tf2_ros
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

def pointcloud_callback(data):
    try:
        trans = tf_buffer.lookup_transform('world', 'velodyne', rospy.Time(0), rospy.Duration(1.0))
        q = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]

        euler_angles = tf_trans.euler_from_quaternion(q)
        roll = euler_angles[0]
        pitch = euler_angles[1]
        counter_rotation = tf_trans.quaternion_from_euler(-roll, -pitch, 0)

        counter_trans = TransformStamped()
        counter_trans.header.stamp = rospy.Time.now()
        counter_trans.header.frame_id = 'velodyne'
        counter_trans.child_frame_id = 'velodyne_horizontal'
        counter_trans.transform.rotation = Quaternion(*counter_rotation)

        broadcaster.sendTransform(counter_trans)
        trans1 = tf_buffer.lookup_transform('velodyne_horizontal', 'velodyne', rospy.Time(0), rospy.Duration(1.0))

        cloud_out = do_transform_cloud(data, trans1)

        cloud_out.header.frame_id = 'velodyne_horizontal'

        velodyne_pub.publish(cloud_out)

    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException, tf2_ros.TransformException) as ex:
        rospy.logerr(ex)


if __name__ == '__main__':
    rospy.init_node('velodyne_horizontal_node', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    broadcaster = tf2_ros.TransformBroadcaster()

    velodyne_pub = rospy.Publisher('velodyne_horizontal_points', PointCloud2, queue_size=10)

    rospy.Subscriber("mid/points", PointCloud2, pointcloud_callback)

    rospy.spin()
