#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros


class TransformFusion(Node):
    def __init__(self):
        super().__init__("transform_fusion")

        self.cur_odom_to_baselink = None
        self.cur_map_to_odom = None

        # Stamp of the last /Odometry we already published downstream, so the
        # 50 Hz timer does not re-broadcast a stale transform as if it were a
        # fresh observation.
        self._last_pub_odom_stamp = None
        # Warn (throttled) once /Odometry lag crosses this, so the pipeline
        # backlog is visible instead of silently feeding Nav2 an old pose.
        self.stale_odom_warn_sec = 0.3

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.pub_localization = self.create_publisher(Odometry, "/localization", 1)

        self.create_subscription(Odometry, "/Odometry", self.cb_save_cur_odom, 1)
        self.create_subscription(Odometry, "/map_to_odom", self.cb_save_map_to_odom, 1)

        self.freq_pub_localization = 50
        self.timer = self.create_timer(1.0 / self.freq_pub_localization, self.transform_fusion)

    def cb_save_cur_odom(self, msg):
        self.cur_odom_to_baselink = msg

    def cb_save_map_to_odom(self, msg):
        self.cur_map_to_odom = msg

    def transform_fusion(self):
        odom_msg = self.cur_odom_to_baselink
        if odom_msg is None:
            return

        # Timestamp of the source odometry (= LiDAR sweep end time). Everything
        # published here is stamped with this, NOT with wall-clock now(), so a
        # stalled FAST-LIO shows up downstream as a TF timeout instead of a
        # confident but seconds-old pose.
        odom_stamp = odom_msg.header.stamp
        odom_stamp_key = (odom_stamp.sec, odom_stamp.nanosec)
        if odom_stamp_key == self._last_pub_odom_stamp:
            # No new odometry since last tick - nothing fresh to assert.
            return
        self._last_pub_odom_stamp = odom_stamp_key

        odom_age = (self.get_clock().now() - Time.from_msg(odom_stamp)).nanoseconds / 1e9
        if odom_age > self.stale_odom_warn_sec:
            self.get_logger().warn(
                f"/Odometry is {odom_age:.2f}s old - localization/TF will be stale",
                throttle_duration_sec=2.0,
            )

        map_to_odom_msg = self.cur_map_to_odom

        # 1. Extract Map -> Odom pose directly (avoid matrix conversion roundtrips)
        if map_to_odom_msg is not None:
            p_m2o = map_to_odom_msg.pose.pose.position
            q_m2o = map_to_odom_msg.pose.pose.orientation
            pos_m2o = np.array([p_m2o.x, p_m2o.y, p_m2o.z])
            rot_m2o = R.from_quat([q_m2o.x, q_m2o.y, q_m2o.z, q_m2o.w])
        else:
            pos_m2o = np.zeros(3)
            rot_m2o = R.identity()

        # 2. Publish TF: map -> camera_init
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = odom_stamp
        tf_stamped.header.frame_id = "map"
        tf_stamped.child_frame_id = "camera_init"

        tf_stamped.transform.translation.x = pos_m2o[0]
        tf_stamped.transform.translation.y = pos_m2o[1]
        tf_stamped.transform.translation.z = pos_m2o[2]

        q_m2o_vec = rot_m2o.as_quat()
        tf_stamped.transform.rotation.x = q_m2o_vec[0]
        tf_stamped.transform.rotation.y = q_m2o_vec[1]
        tf_stamped.transform.rotation.z = q_m2o_vec[2]
        tf_stamped.transform.rotation.w = q_m2o_vec[3]

        self.tf_broadcaster.sendTransform(tf_stamped)

        # 3. Fast Vector Composition: T_map_to_base = T_map_to_odom * T_odom_to_base
        p_o2b = odom_msg.pose.pose.position
        q_o2b = odom_msg.pose.pose.orientation
        pos_o2b = np.array([p_o2b.x, p_o2b.y, p_o2b.z])
        rot_o2b = R.from_quat([q_o2b.x, q_o2b.y, q_o2b.z, q_o2b.w])

        # p_m2b = R_m2o * p_o2b + p_m2o
        pos_m2b = rot_m2o.apply(pos_o2b) + pos_m2o
        # R_m2b = R_m2o * R_o2b
        rot_m2b = rot_m2o * rot_o2b
        q_m2b_vec = rot_m2b.as_quat()

        # 4. Construct and publish localization msg
        localization = Odometry()
        localization.header.stamp = odom_stamp
        localization.header.frame_id = "map"
        localization.child_frame_id = "body"

        localization.pose.pose = Pose(
            position=Point(x=pos_m2b[0], y=pos_m2b[1], z=pos_m2b[2]),
            orientation=Quaternion(
                x=q_m2b_vec[0], y=q_m2b_vec[1], z=q_m2b_vec[2], w=q_m2b_vec[3]
            ),
        )
        localization.twist = odom_msg.twist

        self.pub_localization.publish(localization)


def main(args=None):
    rclpy.init(args=args)
    node = TransformFusion()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
