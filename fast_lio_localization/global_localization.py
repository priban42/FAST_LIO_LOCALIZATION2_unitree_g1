#!/usr/bin/env python3

import copy
import math

import numpy as np
import open3d as o3d
import open3d.core as o3c
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
import tf_transformations
import ros2_numpy

try:  # installed next to this script; needs small_gicp (pip3 install small_gicp)
    import fast_global_localizer
    _FAST_IMPORT_ERROR = None
except ImportError as e:
    fast_global_localizer = None
    _FAST_IMPORT_ERROR = e


def quat_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])


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


class FastLIOLocalization(Node):
    def __init__(self):
        super().__init__("fast_lio_localization")
        self.global_map = None
        self.global_map_colors = None
        self.T_map_to_odom = np.eye(4)
        self.initialized = False

        # One-shot capturing state variables
        self._processing_lock = False
        self._pending_initial_pose_msg = None
        self._captured_scan_msg = None
        self._captured_odom_msg = None
        self._scan_sub = None
        self._odom_sub = None

        self._acc_samples = []
        self._gravity_aligned = False
        self.device = o3c.Device("CPU:0")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_voxel_size", 0.4),
                ("scan_voxel_size", 0.1),
                ("freq_global_map", 0.25),
                ("localization_threshold", 0.8),
                ("fov", 6.28319),
                ("fov_far", 300),
                ("pcd_map_topic", "/map"),
                ("pcd_map_path", ""),
                ("imu_topic", "/livox/imu"),
                ("num_samples", 100),
                ("base_roll", 0.0),
                ("base_pitch", 0.0),
                ("base_yaw", 0.0),
                ("icp_scales", [5.0, 1.0]),
                ("yaw_retry_fitness_threshold", 0.95),
                ("yaw_retry_max_attempts", 4),
                ("translation_retry_step", 7.0),
                ("translation_retry_max_attempts", 9),
                ("use_tensor_api", True),
                # 'fast_gicp': multi-hypothesis VGICP (fast_global_localizer.py) on a few merged frames.
                # 'icp': the point-to-point ICP with yaw / translation retries below.
                ("localization_method", "fast_gicp"),
                ("fast_search_radius", 6.0),      # m around the initial pose
                ("fast_scan_frames", 10),         # /cloud_registered frames merged (10 = ~1 s)
                ("fast_num_threads", 4),
            ],
        )

        self.pub_map_to_odom = self.create_publisher(Odometry, "/map_to_odom", 10)

        self.method = str(self.get_parameter("localization_method").value).strip().lower()
        if self.method not in ("fast_gicp", "icp"):
            self.get_logger().error(f"Unknown localization_method '{self.method}', using 'fast_gicp'.")
            self.method = "fast_gicp"
        self.fast_localizer = None
        self._raw_map_xyz = None
        self._fast_clouds = []
        self._fast_odoms = []

        self.get_logger().info("Waiting for global map...")
        self.initialize_global_map()
        self.get_logger().info("Global map received.")
        if self.method == "fast_gicp":
            self._init_fast_localizer()
        self.get_logger().info(f"Global localization method: {self.method}")

        # Always listen for initialpose command
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.cb_initialize_pose, 10)

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

    # ------------------------------------------------------------------
    # Gravity Alignment
    # ------------------------------------------------------------------
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
            self.get_logger().error("Mean accelerometer norm is ~0. Can't calculate gravity. Retrying...")
            self._acc_samples = []
            return

        z_meas = mean_acc / norm
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
            f"Estimated tilt: roll={roll_est:.2f} deg, pitch={pitch_est:.2f} deg."
        )

        self.T_map_to_odom = np.eye(4)
        self.T_map_to_odom[:3, :3] = tf_transformations.quaternion_matrix(q_total)[:3, :3]

        self._gravity_aligned = True
        self.initialized = True
        self.get_logger().info("Gravity Alignment Success! System now idle and waiting for initialpose.")

        self.destroy_subscription(self._imu_sub)

    # ------------------------------------------------------------------
    # On-Demand Single-Frame Capture Trigger
    # ------------------------------------------------------------------
    def cb_initialize_pose(self, msg: PoseWithCovarianceStamped):
        if not self.initialized:
            self.get_logger().warn("System not yet fully initialized/gravity aligned. Ignoring initialpose.")
            return

        if self._processing_lock:
            self.get_logger().warn("Localization already in progress. Ignoring concurrent initialpose request.")
            return

        self._processing_lock = True
        self._pending_initial_pose_msg = msg
        self._captured_scan_msg = None
        self._captured_odom_msg = None

        if self.method == "fast_gicp":
            self._fast_clouds = []
            self._fast_odoms = []
            n = self.get_parameter("fast_scan_frames").value
            self.get_logger().info(f"Initial pose received. Collecting {n} frames of /cloud_registered and /Odometry...")
            self._scan_sub = self.create_subscription(PointCloud2, "/cloud_registered", self._cb_fast_scan, 10)
            self._odom_sub = self.create_subscription(Odometry, "/Odometry", self._cb_fast_odom, 10)
            return

        self.get_logger().info("Initial pose received. Subscribing to /cloud_registered and /Odometry for single frame capture...")

        self._scan_sub = self.create_subscription(PointCloud2, "/cloud_registered", self._cb_one_shot_scan, 10)
        self._odom_sub = self.create_subscription(Odometry, "/Odometry", self._cb_one_shot_odom, 10)

    def _cb_one_shot_scan(self, msg: PointCloud2):
        if self._captured_scan_msg is None:
            self._captured_scan_msg = msg
            self._check_and_execute_localization()

    def _cb_one_shot_odom(self, msg: Odometry):
        if self._captured_odom_msg is None:
            self._captured_odom_msg = msg
            self._check_and_execute_localization()

    def _check_and_execute_localization(self):
        if self._captured_scan_msg is not None and self._captured_odom_msg is not None:
            # IMMEDIATELY UNSUBSCRIBE TO FREE CPU
            if self._scan_sub is not None:
                self.destroy_subscription(self._scan_sub)
                self._scan_sub = None
            if self._odom_sub is not None:
                self.destroy_subscription(self._odom_sub)
                self._odom_sub = None

            self.get_logger().info("Frame pair captured. Topics unsubscribed. Computing alignment...")

            scan_msg = self._captured_scan_msg
            odom_msg = self._captured_odom_msg
            pose_msg = self._pending_initial_pose_msg

            xyz, intensity = self.msg_to_array(scan_msg)
            cur_scan = o3d.geometry.PointCloud()
            cur_scan.points = o3d.utility.Vector3dVector(xyz)

            initial_pose_mat = self.pose_to_mat(pose_msg.pose.pose)
            T_upside_down = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            initial_pose_mat = initial_pose_mat @ T_upside_down

            T_odom_to_base_link = self.pose_to_mat(odom_msg.pose.pose)
            pose_estimation = np.matmul(initial_pose_mat, self.inverse_se3(T_odom_to_base_link))

            # Execute global localization calculation
            self.global_localization(
                pose_estimation,
                is_initialpose=True,
                cur_scan=cur_scan,
                scan_intensity=intensity,
                cur_odom=odom_msg
            )

            # Reset internal capture state and unlock
            self._pending_initial_pose_msg = None
            self._captured_scan_msg = None
            self._captured_odom_msg = None
            self._processing_lock = False

            self.get_logger().info("Localization step completed. System returned to idle state.")

    # ------------------------------------------------------------------
    # Fast multi-hypothesis GICP (localization_method: fast_gicp)
    # ------------------------------------------------------------------
    def _init_fast_localizer(self):
        if fast_global_localizer is None:
            self.get_logger().error(
                f"localization_method 'fast_gicp' needs small_gicp ({_FAST_IMPORT_ERROR}). "
                "Install it with: pip3 install small_gicp. Falling back to 'icp'.")
            self.method = "icp"
            return
        try:
            params = {
                "radius": float(self.get_parameter("fast_search_radius").value),
                "num_threads": int(self.get_parameter("fast_num_threads").value),
            }
            self.fast_localizer = fast_global_localizer.FastGlobalLocalizer(self._raw_map_xyz, params)
            self.get_logger().info(
                f"fast_gicp ready: {self.fast_localizer.map_points} map points, search radius "
                f"{params['radius']:.1f} m, prepared in {self.fast_localizer.build_time:.2f} s.")
        except Exception as e:  # noqa: BLE001 - never leave the node without a localization method
            self.get_logger().error(f"fast_gicp could not be prepared ({type(e).__name__}: {e}). Falling back to 'icp'.")
            self.fast_localizer = None
            self.method = "icp"
        finally:
            self._raw_map_xyz = None

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _cb_fast_scan(self, msg: PointCloud2):
        if self._pending_initial_pose_msg is None:
            return
        xyz, _ = self.msg_to_array(msg)
        self._fast_clouds.append((self._stamp(msg), np.asarray(xyz, dtype=np.float64).reshape(-1, 3), msg))
        n = max(1, int(self.get_parameter("fast_scan_frames").value))
        del self._fast_clouds[:-n]
        self._fast_try_localize()

    def _cb_fast_odom(self, msg: Odometry):
        if self._pending_initial_pose_msg is None:
            return
        self._fast_odoms.append((self._stamp(msg), msg))
        del self._fast_odoms[:-100]
        self._fast_try_localize()

    def _fast_try_localize(self):
        n = max(1, int(self.get_parameter("fast_scan_frames").value))
        if len(self._fast_clouds) < n or not self._fast_odoms:
            return
        # FAST-LIO stamps a frame and its odometry identically: use the odometry of the newest frame.
        stamp = self._fast_clouds[-1][0]
        odom = next((m for s, m in reversed(self._fast_odoms) if abs(s - stamp) < 0.005), None)
        if odom is None:
            if self._fast_odoms[-1][0] < stamp + 0.3:
                return   # its odometry may still come
            odom = min(self._fast_odoms, key=lambda so: abs(so[0] - stamp))[1]
            self.get_logger().warn("No odometry with the stamp of the newest frame, using the closest one.")

        for attr in ("_scan_sub", "_odom_sub"):
            if getattr(self, attr) is not None:
                self.destroy_subscription(getattr(self, attr))
                setattr(self, attr, None)
        clouds = self._fast_clouds
        pose_msg = self._pending_initial_pose_msg
        self._fast_clouds, self._fast_odoms = [], []
        try:
            self._fast_localization(pose_msg, clouds, odom)
        finally:
            self._pending_initial_pose_msg = None
            self._processing_lock = False
            self.get_logger().info("Localization step completed. System returned to idle state.")

    def _fast_guess(self, initial_pose_mat, T_odom_to_base_link):
        """Initial map->body estimate. An initial pose without roll/pitch (the map view sends x, y, yaw)
        gets them from gravity: the current map->odom (IMU gravity alignment or the previous localization)
        with the odometry; only its heading is replaced by the initial pose's. Poses with roll/pitch
        (e.g. from localization_monitor) are used as they are."""
        guess = initial_pose_mat.copy()
        if abs(guess[2, 2]) < math.cos(math.radians(0.5)):
            return guess, "initial pose"
        prior = self.T_map_to_odom @ T_odom_to_base_link
        dyaw = math.atan2(guess[1, 0], guess[0, 0]) - math.atan2(prior[1, 0], prior[0, 0])
        c, s = math.cos(dyaw), math.sin(dyaw)
        guess[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ prior[:3, :3]
        return guess, "gravity"

    def _fast_localization(self, pose_msg, clouds, odom_msg):
        T_odom_to_base_link = self.pose_to_mat(odom_msg.pose.pose)
        T_base_link_to_odom = self.inverse_se3(T_odom_to_base_link)
        T_upside_down = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        initial_pose_mat = self.pose_to_mat(pose_msg.pose.pose) @ T_upside_down
        guess, tilt_from = self._fast_guess(initial_pose_mat, T_odom_to_base_link)

        merged = np.vstack([c[1] for c in clouds])     # camera_init (odom) frame
        cloud_body = merged @ T_base_link_to_odom[:3, :3].T + T_base_link_to_odom[:3, 3]
        self.get_logger().info(
            f"fast_gicp: {len(clouds)} frames, {len(cloud_body)} points, roll/pitch from {tilt_from}. Computing alignment...")
        try:
            T_map_body, score, info = self.fast_localizer.localize(cloud_body, guess)
        except Exception as e:  # noqa: BLE001 - fall back to the previous method for this request
            self.get_logger().error(f"fast_gicp failed ({type(e).__name__}: {e}). Using 'icp' for this initial pose.")
            cur_scan = o3d.geometry.PointCloud()
            cur_scan.points = o3d.utility.Vector3dVector(clouds[-1][1])
            pose_estimation = np.matmul(initial_pose_mat, T_base_link_to_odom)
            self.global_localization(pose_estimation, is_initialpose=True, cur_scan=cur_scan, cur_odom=odom_msg)
            return

        transformation = T_map_body @ T_base_link_to_odom
        d = T_map_body[:3, 3] - guess[:3, 3]
        dyaw = math.degrees(math.atan2(T_map_body[1, 0], T_map_body[0, 0]) - math.atan2(guess[1, 0], guess[0, 0]))
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        self.get_logger().info(
            f"fast_gicp: score {score:.3f} (fraction within 0.3 m of the map), moved {math.hypot(d[0], d[1]):.2f} m "
            f"and {dyaw:.1f} deg from the initial pose, {info['hypotheses']} hypotheses, {info['refined']} refined, "
            f"{info['time'] * 1000:.0f} ms.")
        self.get_logger().info(f"Applying transformation matrix (Fitness: {score:.4f})")
        self.T_map_to_odom = transformation
        self.publish_odom(transformation)

    # ------------------------------------------------------------------
    # Helper Functions & ICP Logic
    # ------------------------------------------------------------------
    def pose_to_mat(self, pose):
        trans = np.eye(4)
        trans[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return trans

    def msg_to_array(self, pc_msg):
        pc_array = ros2_numpy.numpify(pc_msg)
        xyz = pc_array["xyz"]
        intensity = None
        if "intensity" in pc_array:
            intensity = np.asarray(pc_array["intensity"]).reshape(-1).astype(np.float32)
        return xyz, intensity

    def registration_legacy(self, scan_down, map_down, initial, scale):
        result_icp = o3d.pipelines.registration.registration_icp(
            scan_down,
            map_down,
            1.0 * scale,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-7, relative_rmse=1e-7),
        )
        return result_icp.transformation, result_icp.fitness

    def registration_tensor(self, scan_tensor, map_tensor, initial, scale):
        init_tensor = o3c.Tensor(initial, dtype=o3c.float64, device=self.device)
        result_icp = o3d.t.pipelines.registration.icp(
            scan_tensor,
            map_tensor,
            max_correspondence_distance=1.0 * scale,
            init_source_to_target=init_tensor,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-7, relative_rmse=1e-7),
        )
        transformation = result_icp.transformation.numpy()
        fitness = float(result_icp.fitness)
        return transformation, fitness

    def _yaw_retry_angle_deg(self, retry_index):
        if retry_index % 2 == 1:
            return 180.0
        return 180.0 / (2 ** (retry_index // 2))

    def _rotate_pose_z(self, pose, angle_rad):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot = np.eye(4)
        rot[:3, :3] = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return np.matmul(pose, rot)

    def _generate_grid_offsets(self, step, max_attempts):
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

            layer_points.sort(key=lambda p: (abs(p[0]) + abs(p[1]), p[0]**2 + p[1]**2))
            offsets.extend(layer_points)
            layer += 1

        return offsets[:max_attempts]

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        trans_inverse[:3, :3] = trans[:3, :3].T
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def crop_global_map_in_FOV(self, pose_estimation, cur_odom):
        T_odom_to_base_link = self.pose_to_mat(cur_odom.pose.pose)
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

        if self.global_map_colors is not None:
            cropped_colors = np.squeeze(self.global_map_colors[indices, :])
            global_map_in_FOV.colors = o3d.utility.Vector3dVector(cropped_colors.astype(np.float64) / 255.0)

        return global_map_in_FOV

    def global_localization(self, pose_estimation, is_initialpose=False, cur_scan=None, scan_intensity=None, cur_odom=None):
        if cur_scan is None or cur_odom is None:
            return

        scan_tobe_mapped = copy.copy(cur_scan)
        global_map_in_FOV = self.crop_global_map_in_FOV(pose_estimation, cur_odom)
        if global_map_in_FOV is None:
            return

        scales = self.get_parameter("icp_scales").value
        retry_threshold = self.get_parameter("yaw_retry_fitness_threshold").value
        max_yaw_attempts = self.get_parameter("yaw_retry_max_attempts").value
        translation_step = self.get_parameter("translation_retry_step").value
        max_trans_attempts = self.get_parameter("translation_retry_max_attempts").value
        use_tensor_api = self.get_parameter("use_tensor_api").value

        scan_voxel_size = self.get_parameter("scan_voxel_size").value
        map_voxel_size = self.get_parameter("map_voxel_size").value

        cached_scans = {}
        cached_maps = {}

        for scale in scales:
            scan_down = self.voxel_down_sample(scan_tobe_mapped, scan_voxel_size * scale)
            map_down = self.voxel_down_sample(global_map_in_FOV, map_voxel_size * scale)

            if use_tensor_api:
                cached_scans[scale] = o3d.t.geometry.PointCloud.from_legacy(scan_down, device=self.device)
                cached_maps[scale] = o3d.t.geometry.PointCloud.from_legacy(map_down, device=self.device)
            else:
                cached_scans[scale] = scan_down
                cached_maps[scale] = map_down

        best_transformation = pose_estimation
        best_fitness = -1.0
        success = False

        grid_offsets = self._generate_grid_offsets(translation_step, max_trans_attempts)

        for grid_idx, (tx, ty) in enumerate(grid_offsets):
            shifted_pose = copy.deepcopy(pose_estimation)
            shifted_pose[0, 3] += tx
            shifted_pose[1, 3] += ty

            current_initial = shifted_pose
            attempt = 1

            while True:
                self.get_logger().info(
                    f"Grid [{grid_idx + 1}/{len(grid_offsets)}]: offset [{tx:.1f}, {ty:.1f}], rot attempt {attempt}/{max_yaw_attempts + 1}..."
                )

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

    def voxel_down_sample(self, pcd, voxel_size):
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        except Exception:
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
        return pcd_down

    def initialize_global_map(self):
        self.global_map = o3d.io.read_point_cloud(self.get_parameter("pcd_map_path").value)
        if self.method == "fast_gicp":   # fast_gicp uses the full-resolution map (it downsamples to 0.1 m)
            self._raw_map_xyz = np.asarray(self.global_map.points).copy()
        self.global_map = self.voxel_down_sample(self.global_map, self.get_parameter("map_voxel_size").value)

        if len(self.global_map.colors) > 0:
            self.global_map_colors = (np.asarray(self.global_map.colors) * 255.0).astype(np.uint8)
        else:
            self.global_map_colors = None

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


def main(args=None):
    rclpy.init(args=args)
    node = FastLIOLocalization()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
