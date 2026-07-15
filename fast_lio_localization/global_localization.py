#!/usr/bin/env python3

import copy
import math
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c  # Open3D Tensor Core
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu
from std_msgs.msg import Header
import tf2_ros
import tf_transformations
import ros2_numpy


# --- Helper Quaternion Math Functions ---

def quat_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])  # x,y,z,w


def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def build_point_cloud2(header, xyz, intensity=None, rgb=None):
    """
    Build a PointCloud2 message directly, without going through
    ros2_numpy's dict_to_point_cloud2.
    """
    n = xyz.shape[0]
    xyz32 = xyz[:, :3].astype(np.float32)

    names = ['x', 'y', 'z']
    formats = [np.float32, np.float32, np.float32]
    offsets = [0, 4, 8]
    next_offset = 12

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    intensity16 = None
    if intensity is not None:
        intensity16 = np.asarray(intensity).reshape(-1).astype(np.uint16)
        names.append('intensity')
        formats.append(np.uint16)
        offsets.append(next_offset)
        fields.append(PointField(name='intensity', offset=next_offset, datatype=PointField.UINT16, count=1))
        next_offset += 2

    rgb_packed = None
    if rgb is not None:
        rgb_u8 = np.asarray(rgb).astype(np.uint32)
        rgb_packed = ((rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | rgb_u8[:, 2]).astype(np.uint32)
        names.append('rgb')
        formats.append(np.uint32)
        offsets.append(next_offset)
        fields.append(PointField(name='rgb', offset=next_offset, datatype=PointField.UINT32, count=1))
        next_offset += 4

    point_step = next_offset

    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': point_step})
    structured = np.zeros(n, dtype=dtype)
    structured['x'] = xyz32[:, 0]
    structured['y'] = xyz32[:, 1]
    structured['z'] = xyz32[:, 2]
    if intensity16 is not None:
        structured['intensity'] = intensity16
    if rgb_packed is not None:
        structured['rgb'] = rgb_packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = bool(np.isfinite(xyz32).all())
    msg.data = structured.tobytes()

    return msg


class FastLIOLocalization(Node):
    def __init__(self):
        super().__init__("fast_lio_localization")
        self.global_map = None
        self.global_map_colors = None      # Nx3 uint8, or None if the map has no color
        self.T_map_to_odom = np.eye(4)
        self.cur_odom = None
        self.cur_scan = None
        self.cur_scan_intensity = None      # Nx1 float32, or None if the scan has no intensity
        self.initialized = False

        # IMU Gravity Align specific attributes
        self._acc_samples = []
        self._gravity_aligned = False

        # Initialize Tensor Core CPU device
        self.device = o3c.Device("CPU:0")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_voxel_size", 0.4),
                ("scan_voxel_size", 0.1),
                ("freq_localization", 0.5),
                ("freq_global_map", 0.25),
                ("localization_threshold", 0.8),
                ("fov", 6.28319),
                ("fov_far", 300),
                ("pcd_map_topic", "/map"),
                ("pcd_map_path", ""),
                # Integrated parameters for gravity alignment
                ("imu_topic", "/livox/imu"),
                ("num_samples", 100),
                ("base_roll", 0.0),
                ("base_pitch", 0.0),  # 180 deg pitch to correct inverted mounting
                ("base_yaw", 0.0),
                # Configurable multi-scale ICP schedule
                ("icp_scales", [5.0, 1.0]),
                # Yaw-retry behavior when fitness is too low
                ("yaw_retry_fitness_threshold", 0.95),
                ("yaw_retry_max_attempts", 4),
                # Configurable translation search grid
                ("translation_retry_step", 7.0),
                ("translation_retry_max_attempts", 9),
                # Pipeline backend selection (Tensor vs Legacy)
                ("use_tensor_api", True),
            ],
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_pc_in_map = self.create_publisher(PointCloud2, "/cur_scan_in_map", 10)
        self.pub_submap = self.create_publisher(PointCloud2, "/submap", 10)
        self.pub_map_to_odom = self.create_publisher(Odometry, "/map_to_odom", 10)

        # New Debug Publisher to verify coordinate frame alignment visually
        self.pub_aligned_debug = self.create_publisher(PointCloud2, "/cloud_aligned_debug", 10)

        self.get_logger().info("Waiting for global map...")
        self.initialize_global_map()
        self.get_logger().info("Global map received.")

        # Subscriptions
        self.create_subscription(PointCloud2, "/cloud_registered", self.cb_save_cur_scan, 10)
        self.create_subscription(Odometry, "/Odometry", self.cb_save_cur_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.cb_initialize_pose, 10)

        # Active gravity alignment subscription
        self.imu_topic = self.get_parameter("imu_topic").value
        self.num_samples = self.get_parameter("num_samples").value

        if self.num_samples > 0:
            self.get_logger().info(
                f"Starting active Gravity Alignment phase. Collecting {self.num_samples} "
                f"samples from '{self.imu_topic}'. KEEP THE ROBOT STATIONARY!"
            )
            self._imu_sub = self.create_subscription(
                Imu, self.imu_topic, self.cb_imu_align, qos_profile_sensor_data
            )
        else:
            self.get_logger().warn("Gravity alignment deactivated (num_samples=0). Waiting for initialpose manual click.")

        self.timer_localisation = self.create_timer(
            1.0 / self.get_parameter("freq_localization").value,
            self.localisation_timer_callback
        )

    def cb_imu_align(self, msg: Imu):
        if self._gravity_aligned:
            return

        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        self._acc_samples.append(acc)

        if len(self._acc_samples) >= self.num_samples:
            self._compute_gravity_alignment()

    def _compute_gravity_alignment(self):
        mean_acc = np.mean(self._acc_samples, axis=0)
        norm = np.linalg.norm(mean_acc)

        if norm < 1e-3:
            self.get_logger().error(
                "Mean accelerometer norm is ~0. Can't calculate gravity. Retrying..."
            )
            self._acc_samples = []
            return

        z_meas = mean_acc / norm  # measured "up" direction in IMU frame
        z_world = np.array([0.0, 0.0, 1.0])

        dot = float(np.clip(np.dot(z_meas, z_world), -1.0, 1.0))
        angle = math.acos(dot)
        axis = np.cross(z_meas, z_world)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            q_tilt = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            q_tilt = quat_from_axis_angle(axis, angle)

        base_roll = self.get_parameter("base_roll").value
        base_pitch = self.get_parameter("base_pitch").value
        base_yaw = self.get_parameter("base_yaw").value

        q_base = quat_from_euler(base_roll, base_pitch, base_yaw)
        q_total = quat_multiply(q_base, q_tilt)

        roll_est = math.degrees(math.atan2(mean_acc[1], mean_acc[2]))
        pitch_est = math.degrees(math.atan2(-mean_acc[0], math.hypot(mean_acc[1], mean_acc[2])))
        self.get_logger().info(
            f"Estimated tilt: roll={roll_est:.2f} deg, pitch={pitch_est:.2f} deg. "
            f"Offsetting mounting tilt parameters (Pitch={math.degrees(base_pitch):.2f})."
        )

        # Convert calculated alignment quaternion directly into self.T_map_to_odom
        self.T_map_to_odom = np.eye(4)
        self.T_map_to_odom[:3, :3] = tf_transformations.quaternion_matrix(q_total)[:3, :3]

        self._gravity_aligned = True
        self.initialized = True
        self.get_logger().info("Gravity Alignment Success! System now armed and localizing.")

        # Clean up and destroy active IMU callback
        self.destroy_subscription(self._imu_sub)

    def pose_to_mat(self, pose):
        trans = np.eye(4)
        trans[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return trans

    def msg_to_array(self, pc_msg):
        """Returns (xyz, intensity_or_None)."""
        pc_array = ros2_numpy.numpify(pc_msg)
        xyz = pc_array["xyz"]
        intensity = None
        if "intensity" in pc_array:
            intensity = np.asarray(pc_array["intensity"]).reshape(-1).astype(np.float32)
        return xyz, intensity

    def registration_legacy(self, scan_down, map_down, initial, scale):
        """Legacy registration pipeline using Open3D's standard API."""
        result_icp = o3d.pipelines.registration.registration_icp(
            scan_down,
            map_down,
            1.0 * scale,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10),
        )
        return result_icp.transformation, result_icp.fitness

    def registration_tensor(self, scan_tensor, map_tensor, initial, scale):
        """High-performance registration pipeline utilizing Open3D Tensors."""
        init_tensor = o3c.Tensor(initial, dtype=o3c.float64, device=self.device)
        result_icp = o3d.t.pipelines.registration.icp(
            scan_tensor,
            map_tensor,
            max_correspondence_distance=1.0 * scale,
            init_source_to_target=init_tensor,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=10),
        )
        # Convert tensor attributes back to python/numpy format
        transformation = result_icp.transformation.numpy()
        fitness = float(result_icp.fitness)
        return transformation, fitness

    def _yaw_retry_angle_deg(self, retry_index):
        """
        retry_index starts at 1 for the first retry (i.e. the 2nd overall attempt).
        Sequence: 180, 90, 180, 45, 180, 22.5, 180, ...
        """
        if retry_index % 2 == 1:
            return 180.0
        return 180.0 / (2 ** (retry_index // 2))

    def _rotate_pose_z(self, pose, angle_rad):
        """Rotate a pose about its own z-axis (yaw), keeping position unchanged."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot = np.eye(4)
        rot[:3, :3] = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return np.matmul(pose, rot)

    def _generate_grid_offsets(self, step, max_attempts):
        """
        Generates grid translation offsets (dx, dy) starting from [0,0]
        and spiraling outwards: [0, 0], [step, 0], [-step, 0], [0, step], [0, -step]...
        """
        offsets = []
        visited = set()
        layer = 0
        
        while len(offsets) < max_attempts:
            layer_points = []
            for dx in range(-layer, layer + 1):
                for dy in range(-layer, layer + 1):
                    if max(abs(dx), abs(dy)) == layer:
                        pt = (float(dx * step), float(dy * step))
                        if pt not in visited:
                            layer_points.append(pt)
                            visited.add(pt)
            
            # Sort layer coordinates so that closer positions are tried first
            layer_points.sort(key=lambda p: (abs(p[0]) + abs(p[1]), p[0]**2 + p[1]**2))
            offsets.extend(layer_points)
            layer += 1
            
        return offsets[:max_attempts]

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        trans_inverse[:3, :3] = trans[:3, :3].T
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def publish_point_cloud(self, publisher, header, xyz, intensity=None, rgb=None):
        msg = build_point_cloud2(header, xyz, intensity=intensity, rgb=rgb)
        publisher.publish(msg)

    def crop_global_map_in_FOV(self, pose_estimation):
        if self.cur_odom is None:
            self.get_logger().warn("No odometry received yet, skipping this localization step.", once=True)
            return None

        T_odom_to_base_link = self.pose_to_mat(self.cur_odom.pose.pose)
        T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
        T_base_link_to_map = self.inverse_se3(T_map_to_base_link)

        global_map_in_map = np.array(self.global_map.points)
        global_map_in_map_h = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
        global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map_h.T).T

        if self.get_parameter("fov").value > 3.14:
            indices = np.where(
                (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        else:
            indices = np.where(
                (global_map_in_base_link[:, 0] > 0)
                & (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )

        cropped_points = np.squeeze(global_map_in_map[indices, :3])

        global_map_in_FOV = o3d.geometry.PointCloud()
        global_map_in_FOV.points = o3d.utility.Vector3dVector(cropped_points)

        cropped_colors = None
        if self.global_map_colors is not None:
            cropped_colors = np.squeeze(self.global_map_colors[indices, :])
            global_map_in_FOV.colors = o3d.utility.Vector3dVector(cropped_colors.astype(np.float64) / 255.0)

        # --- FIXED LINES ---
        header = copy.deepcopy(self.cur_odom.header)
        header.frame_id = "map"
        # -------------------

        sub_points = np.array(global_map_in_FOV.points)[::10]
        sub_rgb = cropped_colors[::10] if cropped_colors is not None else None
        self.publish_point_cloud(self.pub_submap, header, sub_points, rgb=sub_rgb)

        return global_map_in_FOV

    def global_localization(self, pose_estimation, is_initialpose=False):
        scan_tobe_mapped = copy.copy(self.cur_scan)
        global_map_in_FOV = self.crop_global_map_in_FOV(pose_estimation)
        if global_map_in_FOV is None:
            return

        # Fetch configurations
        scales = self.get_parameter("icp_scales").value
        retry_threshold = self.get_parameter("yaw_retry_fitness_threshold").value
        max_yaw_attempts = self.get_parameter("yaw_retry_max_attempts").value
        translation_step = self.get_parameter("translation_retry_step").value
        max_trans_attempts = self.get_parameter("translation_retry_max_attempts").value
        use_tensor_api = self.get_parameter("use_tensor_api").value

        # Local variables to cache parameter values (avoid ROS 2 parameter overhead in tight loops)
        scan_voxel_size = self.get_parameter("scan_voxel_size").value
        map_voxel_size = self.get_parameter("map_voxel_size").value

        # --- OPTIMIZATION STEP: PRE-DOWNSAMPLE AND CACHE CLOUDS ONCE ---
        cached_scans = {}
        cached_maps = {}

        for scale in scales:
            # 1. Downsample legacy representations
            scan_down = self.voxel_down_sample(scan_tobe_mapped, scan_voxel_size * scale)
            map_down = self.voxel_down_sample(global_map_in_FOV, map_voxel_size * scale)
            
            # 2. Cache either legacy or high-performance tensor formats
            if use_tensor_api:
                cached_scans[scale] = o3d.t.geometry.PointCloud.from_legacy(scan_down, device=self.device)
                cached_maps[scale] = o3d.t.geometry.PointCloud.from_legacy(map_down, device=self.device)
            else:
                cached_scans[scale] = scan_down
                cached_maps[scale] = map_down

        best_transformation = None
        best_fitness = -1.0
        success = False

        grid_offsets = self._generate_grid_offsets(translation_step, max_trans_attempts)

        # Loop through search grid translation offsets
        for grid_idx, (tx, ty) in enumerate(grid_offsets):
            shifted_pose = copy.deepcopy(pose_estimation)
            shifted_pose[0, 3] += tx
            shifted_pose[1, 3] += ty

            current_initial = shifted_pose
            attempt = 1

            # Run orientation retries for this translation offset
            while True:
                self.get_logger().info(
                    f"Grid [{grid_idx + 1}/{len(grid_offsets)}]: offset [{tx:.1f}, {ty:.1f}], rot attempt {attempt}/{max_yaw_attempts + 1}..."
                )
                
                # Manual multi-scale iteration over cached elements
                transformation = current_initial
                fitness = 0.0
                fitness_log = []

                for scale in scales:
                    if use_tensor_api:
                        transformation, fitness = self.registration_tensor(
                            cached_scans[scale], cached_maps[scale], initial=transformation, scale=scale
                        )
                    else:
                        transformation, fitness = self.registration_legacy(
                            cached_scans[scale], cached_maps[scale], initial=transformation, scale=scale
                        )
                    fitness_log.append(fitness)

                self.get_logger().info(
                    f"Offset [{tx:.1f}, {ty:.1f}], Rot Attempt {attempt}: "
                    + ", ".join(f"scale={s:.2f}->fitness={f:.4f}" for s, f in zip(scales, fitness_log))
                )

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_transformation = transformation

                if fitness >= retry_threshold:
                    success = True
                    break

                if attempt > max_yaw_attempts:
                    break

                # Setup next yaw attempt
                retry_index = attempt
                yaw_offset_deg = self._yaw_retry_angle_deg(retry_index)
                current_initial = self._rotate_pose_z(shifted_pose, math.radians(yaw_offset_deg))
                attempt += 1

            if success:
                self.get_logger().info(f"Target fitness {fitness:.4f} achieved at offset [{tx:.1f}, {ty:.1f}]!")
                break

        transformation = best_transformation
        fitness = best_fitness

        if is_initialpose or fitness > self.get_parameter("localization_threshold").value:
            self.get_logger().info(f"Applying transformation matrix (Fitness: {fitness:.4f})")
            self.T_map_to_odom = transformation
            self.publish_odom(transformation)
        else:
            self.get_logger().warn(
                f"Best fitness {fitness:.4f} across all search iterations still below threshold. "
                "Tracking dropped to maintain last safe pose."
            )

        # --- DEBUG VISUALIZATION PIPELINE ---
        scan_aligned = copy.deepcopy(scan_tobe_mapped)
        scan_aligned.transform(transformation)
        aligned_np = np.asarray(scan_aligned.points)

        debug_header = Header()
        debug_header.stamp = self.get_clock().now().to_msg()
        debug_header.frame_id = "map"

        self.publish_point_cloud(
            self.pub_aligned_debug,
            debug_header,
            aligned_np,
            intensity=self.cur_scan_intensity
        )

    def voxel_down_sample(self, pcd, voxel_size):
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        except Exception:
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
        return pcd_down

    def cb_save_cur_odom(self, msg):
        self.cur_odom = msg

    def cb_save_cur_scan(self, msg):
        xyz, intensity = self.msg_to_array(msg)
        self.cur_scan = o3d.geometry.PointCloud()
        self.cur_scan.points = o3d.utility.Vector3dVector(xyz)
        self.cur_scan_intensity = intensity
        self.publish_point_cloud(self.pub_pc_in_map, msg.header, xyz, intensity=intensity)

    def initialize_global_map(self):
        self.global_map = o3d.io.read_point_cloud(self.get_parameter("pcd_map_path").value)
        self.global_map = self.voxel_down_sample(self.global_map, self.get_parameter("map_voxel_size").value)

        if len(self.global_map.colors) > 0:
            self.global_map_colors = (np.asarray(self.global_map.colors) * 255.0).astype(np.uint8)
        else:
            self.global_map_colors = None
            self.get_logger().warn(
                "Global map has no readable color (PCD 'rgb' field must be a packed float32 "
                "for Open3D to load it as color). Submap will be published without color."
            )

    def cb_initialize_pose(self, msg):
        initial_pose = self.pose_to_mat(msg.pose.pose)
        T_upside_down = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        initial_pose = initial_pose@T_upside_down
        self.initialized = True
        self.get_logger().info("Initial pose overridden manually.")

        if self.cur_scan is not None:
            self.global_localization(initial_pose, is_initialpose=True)

    def publish_odom(self, transform):
        odom_msg = Odometry()
        xyz = transform[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(transform)
        odom_msg.pose.pose = Pose(
            position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        )
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        self.pub_map_to_odom.publish(odom_msg)

    def localisation_timer_callback(self):
        if not self.initialized:
            return

        if self.cur_scan is not None:
            self.global_localization(self.T_map_to_odom, is_initialpose=False)


def main(args=None):
    rclpy.init(args=args)
    node = FastLIOLocalization()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
