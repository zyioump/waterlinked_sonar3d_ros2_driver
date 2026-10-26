#!/usr/bin/env python3

import dataclasses
import json
import math
import socket

import cv2 as cv
import numpy as np
import rclpy
import requests
import wlsonar
import wlsonar.range_image_protocol as rip
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger

# Errors raised by the wlsonar HTTP client. RuntimeError is raised by the
# Sonar3D constructor when the sonar cannot be reached.
SONAR_ERRORS = (
    requests.exceptions.RequestException,
    RuntimeError,
    ValueError,
    KeyError,
)


class Sonar3d_driver(Node):
    DOWNSAMPLING = 1
    THRESHOLD = 0

    def __init__(self):
        super().__init__("sonar3d_driver")

        self.declare_parameter("sonar.ip", "192.168.2.199")
        self.declare_parameter("health.check_interval", 3.0)
        self.declare_parameter("health.request_timeout", 1.0)

        self.sonar_ip = self.get_parameter("sonar.ip").value
        self.health_check_interval = self.get_parameter(
            "health.check_interval"
        ).value
        self.health_request_timeout = self.get_parameter(
            "health.request_timeout"
        ).value
        self.interface_ip = self.get_interface_ip(self.sonar_ip)
        # Created lazily so the driver can start while the sonar is down
        self.sonar = None

        log_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.log_pub = self.create_publisher(Log, "/roller/logs", log_qos)
        self.health_log_reported = {
            "sonar_unavailable": False,
            "acoustics_disabled": False,
            "time_unsynchronized": False,
            "rip1_protocol": False,
        }
        self.desired_acoustics_enabled = None

        self.range_pub = self.create_publisher(Image, "sonar3d/range", 1)
        self.range_ui_pub = self.create_publisher(
            CompressedImage, "sonar3d/range/ui/compressed", 1
        )
        self.int_pub = self.create_publisher(Image, "sonar3d/intensity", 1)
        self.int_ui_pub = self.create_publisher(
            CompressedImage, "sonar3d/intensity/ui/compressed", 1
        )
        self.range_int_pub = self.create_publisher(Image, "sonar3d/range_intensity", 1)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "sonar3d/pointcloud", 1
        )
        # IMU samples arrive in batches, keep enough history for a whole batch
        self.imu_pub = self.create_publisher(Imu, "sonar3d/imu", 100)

        self.start_srv = self.create_service(Trigger, "sonar3d/start", self.start_sonar)
        self.stop_srv = self.create_service(Trigger, "sonar3d/stop", self.stop_sonar)
        self.status_srv = self.create_service(
            Trigger, "sonar3d/status", self.get_status
        )
        self.ip_srv = self.create_service(
            Trigger, "sonar3d/get_ip", self.get_sonar_ip
        )

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.01, self.loop)
        self.health_timer = self.create_timer(
            self.health_check_interval,
            self.check_health,
        )

        self.sock = wlsonar.open_sonar_udp_multicast_socket(
            iface_ip=self.interface_ip
        )
        self.sock.settimeout(0.2)

        self.image_sync = []

        self.get_logger().info(
            f"Sonar3d driver started for {self.sonar_ip} "
            f"via interface {self.interface_ip}"
        )

    def get_interface_ip(self, peer_ip):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_sock:
                route_sock.connect((peer_ip, wlsonar.DEFAULT_MCAST_PORT))
                return route_sock.getsockname()[0]
        except OSError as exc:
            self.get_logger().error(
                f"Could not determine local interface IP for sonar: {exc}"
            )
            raise RuntimeError("Sonar3d is not available") from None

    def get_sonar(self):
        if self.sonar is None:
            sonar = wlsonar.Sonar3D(
                self.sonar_ip, timeout=self.health_request_timeout
            )
            # Send images to the multicast group the driver listens on
            sonar.set_udp_multicast()
            self.get_logger().info("Set Sonar3d UDP output to multicast")
            self.sonar = sonar
        return self.sonar

    def set_acoustics(self, value):
        self.get_sonar().set_acoustics_enabled(value)

    def publish_log(self, level, message):
        log_message = Log()
        log_message.stamp = self.get_clock().now().to_msg()
        log_message.level = level
        log_message.name = self.get_name()
        log_message.msg = message
        self.log_pub.publish(log_message)

    def publish_log_once(self, condition, level, message):
        if self.health_log_reported[condition]:
            return

        self.publish_log(level, message)
        self.health_log_reported[condition] = True

    def publish_recovery_log(self, condition, message):
        if not self.health_log_reported[condition]:
            return

        self.publish_log(LoggingSeverity.INFO.value, message)
        self.health_log_reported[condition] = False

    def check_health(self):
        try:
            self.get_sonar().get_status()
            self.publish_recovery_log(
                "sonar_unavailable",
                "Sonar3d is available",
            )
        except SONAR_ERRORS as exc:
            self.get_logger().error(f"Sonar3d status request failed: {exc}")
            # Recreate the client on recovery to refresh the sonar version
            self.sonar = None
            self.health_log_reported["acoustics_disabled"] = False
            self.health_log_reported["time_unsynchronized"] = False
            self.publish_log_once(
                "sonar_unavailable",
                LoggingSeverity.ERROR.value,
                "Sonar3d is not available",
            )
            return

        try:
            time_status = self.sonar.get_time_status()
            if not isinstance(time_status.ntp_synced, bool):
                raise ValueError("invalid time status response")

            if time_status.ntp_synced:
                self.publish_recovery_log(
                    "time_unsynchronized",
                    "Sonar3d time is synchronized",
                )
            else:
                self.publish_log_once(
                    "time_unsynchronized",
                    LoggingSeverity.ERROR.value,
                    "Sonar3d time is not synchronized",
                )
        except SONAR_ERRORS as exc:
            self.get_logger().error(
                f"Sonar3d time status request failed: {exc}"
            )

        self.ensure_imu_output()

        try:
            acoustics_enabled = self.sonar.get_acoustics_enabled()
        except SONAR_ERRORS as exc:
            self.get_logger().error(
                f"Sonar3d acoustics status request failed: {exc}"
            )
            return

        if self.desired_acoustics_enabled is None:
            self.desired_acoustics_enabled = acoustics_enabled

        if self.desired_acoustics_enabled and not acoustics_enabled:
            try:
                self.set_acoustics(True)
                self.publish_recovery_log(
                    "acoustics_disabled",
                    "Sonar3d acoustics are enabled",
                )
                self.get_logger().info(
                    "Re-enabled Sonar3d acoustics after sonar restart"
                )
                return
            except SONAR_ERRORS as exc:
                self.get_logger().error(
                    f"Failed to re-enable Sonar3d acoustics: {exc}"
                )

        if not acoustics_enabled:
            self.publish_log_once(
                "acoustics_disabled",
                LoggingSeverity.WARN.value,
                "Sonar3d acoustics are disabled",
            )
            return

        self.publish_recovery_log(
            "acoustics_disabled",
            "Sonar3d acoustics are enabled",
        )

    def ensure_imu_output(self):
        try:
            if not self.sonar.get_output_imu_batch_enabled():
                self.sonar.set_output_imu_batch_enabled(True)
                self.get_logger().info("Enabled Sonar3d IMU output")
        except SONAR_ERRORS as exc:
            self.get_logger().error(f"Failed to enable Sonar3d IMU output: {exc}")

    def start_sonar(self, req, res):
        self.get_logger().info("Start pinging")
        self.desired_acoustics_enabled = True
        try:
            self.set_acoustics(True)
            res.success = True
            self.publish_recovery_log(
                "acoustics_disabled",
                "Sonar3d acoustics are enabled",
            )
        except SONAR_ERRORS as exc:
            self.get_logger().error(f"Failed to start Sonar3d acoustics: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def stop_sonar(self, req, res):
        self.get_logger().info("Stop pinging")
        self.desired_acoustics_enabled = False
        try:
            self.set_acoustics(False)
            res.success = True
        except SONAR_ERRORS as exc:
            self.get_logger().error(f"Failed to stop Sonar3d acoustics: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def get_status(self, req, res):
        try:
            status = self.get_sonar().get_status()
            res.success = True
            res.message = json.dumps(dataclasses.asdict(status), indent=2)
        except SONAR_ERRORS as exc:
            self.get_logger().error(f"Sonar3d status request failed: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def get_sonar_ip(self, req, res):
        res.success = True
        res.message = self.sonar_ip
        return res

    def to_np(self, img_obj, dtype):
        img = np.array(list(img_obj.image_pixel_data), dtype=dtype)
        img = img.reshape((img_obj.height, img_obj.width))
        return np.flip(img, 0)

    def pair2pc(self, range_img, int_img):
        range_img = np.flip(range_img, 1)
        int_img = np.flip(int_img, 1)
        pc = []
        for i in range(0, range_img.shape[0], self.DOWNSAMPLING):
            for j in range(0, range_img.shape[1], self.DOWNSAMPLING):
                radius = range_img[i, j]
                if radius == 0:
                    continue
                if int_img[i, j] < self.THRESHOLD:
                    continue
                yaw = (j / (range_img.shape[1] - 1)) * self.HFOV - (self.HFOV / 2)
                pitch = (i / (range_img.shape[0] - 1)) * self.VFOV - (self.VFOV / 2)
                x = radius * math.cos(pitch) * math.cos(yaw)
                y = radius * math.cos(pitch) * math.sin(yaw)
                z = -radius * math.sin(pitch)
                pc.append((x, y, z, int_img[i, j]))

        header = self.get_header()

        if len(pc) == 0:
            msg = PointCloud2()
            msg.header = header
            return msg

        pc = np.array(pc)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]

        msg = point_cloud2.create_cloud(header, fields, pc)

        return msg

    def get_header(self):
        # The timestamp in the soanr3d msg is not sync by chrony
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "sonar3d"
        return header

    def loop(self):
        try:
            data, address = self.sock.recvfrom(wlsonar.UDP_MAX_DATAGRAM_SIZE)
        except TimeoutError:
            return
        if address[0] != self.sonar_ip:
            return

        try:
            msg_obj = rip.unpackb(data)
        except rip.UnknownProtobufTypeError:
            # The sonar also sends undocumented internal messages
            return
        except ValueError as exc:
            self.get_logger().warning(f"Invalid Sonar3d packet: {exc}")
            return

        if data[:4] == rip.ProtocolVersion.RIP1.bytes():
            message = "Sonar3d is sending RIP1 packets, RIP1 is deprecated"
            self.get_logger().warning(message, once=True)
            self.publish_log_once(
                "rip1_protocol", LoggingSeverity.WARN.value, message
            )

        if isinstance(msg_obj, rip.ImuBatch):
            self.publish_imu_batch(msg_obj)
            return

        if not isinstance(msg_obj, (rip.BitmapImageGreyscale8, rip.RangeImage)):
            return

        self.HFOV = math.radians(msg_obj.fov_horizontal)
        self.VFOV = math.radians(msg_obj.fov_vertical)
        self.MAX_RANGE = msg_obj.range

        if isinstance(msg_obj, rip.BitmapImageGreyscale8):
            img = self.to_np(msg_obj, np.uint8)
            self.publish_img(
                img, img, self.int_pub, self.int_ui_pub, msg_obj.header.timestamp
            )

            if len(self.image_sync) == 0:
                return
            if self.image_sync[0] == msg_obj.header.sequence_id:
                self.process_img_pair(self.image_sync[1], img, msg_obj.header.timestamp)

        elif isinstance(msg_obj, rip.RangeImage):
            img = self.to_np(msg_obj, np.uint32).astype(float)
            img *= msg_obj.image_pixel_scale
            self.publish_img(
                img,
                (255 * img / self.MAX_RANGE).astype(np.uint8),
                self.range_pub,
                self.range_ui_pub,
                msg_obj.header.timestamp,
            )

            # Range message are received first
            self.image_sync = [msg_obj.header.sequence_id, img]

    def process_img_pair(self, range_img, int_img, timestamp):
        pc = self.pair2pc(range_img, int_img)
        pc.header.stamp.sec = timestamp.seconds
        pc.header.stamp.nanosec = timestamp.nanos
        self.pointcloud_pub.publish(pc)

        new_shape = list(range_img.shape)
        new_shape.append(1)

        range_int_img = np.concatenate(
            (
                range_img.reshape(new_shape),
                int_img.reshape(new_shape),
                np.zeros(new_shape),
            ),
            axis=2,
        )
        msg = self.bridge.cv2_to_imgmsg(range_int_img)
        msg.header = pc.header

        self.range_int_pub.publish(msg)

    def publish_img(self, img, img_ui, pub, pub_ui, timestamp):
        msg = self.bridge.cv2_to_imgmsg(img)
        msg.header = self.get_header()
        msg.header.stamp.sec = timestamp.seconds
        msg.header.stamp.nanosec = timestamp.nanos
        img_ui = cv.applyColorMap(img_ui, cv.COLORMAP_JET)
        img_ui[img == 0] = (0, 0, 0)
        img_ui = cv.resize(
            img_ui, (int(img_ui.shape[1] * self.HFOV / self.VFOV), img_ui.shape[1])
        )
        msg_ui = self.bridge.cv2_to_compressed_imgmsg(img_ui)
        msg_ui.header = msg.header
        msg_ui.header.stamp.sec = timestamp.seconds
        msg_ui.header.stamp.nanosec = timestamp.nanos
        pub.publish(msg)
        pub_ui.publish(msg_ui)

    def publish_imu_batch(self, batch):
        if (
            len(batch.timestamp) != batch.samples
            or len(batch.specific_force) != 3 * batch.samples
            or len(batch.rate_of_turn) != 3 * batch.samples
        ):
            self.get_logger().warning(
                f"Invalid Sonar3d IMU batch {batch.batch_sequence_id}"
            )
            return

        for k in range(batch.samples):
            msg = Imu()
            msg.header = self.get_header()
            msg.header.stamp.sec = batch.timestamp[k].seconds
            msg.header.stamp.nanosec = batch.timestamp[k].nanos
            # The sonar does not provide orientation (REP 145)
            msg.orientation_covariance[0] = -1.0
            # Rotate 180 deg about x: sonar IMU is z down, ROS is z up
            fx, fy, fz = batch.specific_force[3 * k : 3 * k + 3]
            msg.linear_acceleration.x = fx
            msg.linear_acceleration.y = -fy
            msg.linear_acceleration.z = -fz
            wx, wy, wz = batch.rate_of_turn[3 * k : 3 * k + 3]
            msg.angular_velocity.x = wx
            msg.angular_velocity.y = -wy
            msg.angular_velocity.z = -wz
            self.imu_pub.publish(msg)

    def __del__(self):
        if hasattr(self, "sock"):
            self.sock.close()


def main(args=None):
    rclpy.init(args=args)
    driver = Sonar3d_driver()

    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        pass

    driver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
