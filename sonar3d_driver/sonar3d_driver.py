#!/usr/bin/env python3

import math
import socket
import struct
import zlib
import snappy

import cv2 as cv
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from sonar3d_driver.sonar_3d_15_protocol_pb2 import (
    BitmapImageGreyscale8,
    Packet,
    RangeImage,
)


class Sonar3d_driver(Node):
    BUFFER_SIZE = 65535
    MULTICAST_GROUP = "224.0.0.96"

    PORT = 4747

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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.PORT))
        self.sock.settimeout(0.2)

        group = socket.inet_aton(self.MULTICAST_GROUP)
        mreq = struct.pack("4s4s", group, socket.inet_aton(self.interface_ip))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        self.image_sync = []

        self.get_logger().info(
            f"Sonar3d driver started for {self.sonar_ip} "
            f"via interface {self.interface_ip}"
        )

    def get_interface_ip(self, peer_ip):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_sock:
                route_sock.connect((peer_ip, self.PORT))
                return route_sock.getsockname()[0]
        except OSError as exc:
            self.get_logger().error(
                f"Could not determine local interface IP for sonar: {exc}"
            )
            raise RuntimeError("Sonar3d is not available") from None

    def set_acoustics(self, value):
        res = requests.post(
            f"http://{self.sonar_ip}/api/v1/integration/acoustics/enabled",
            json=value,
            timeout=5,
        )
        return res

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
        status_url = f"http://{self.sonar_ip}/api/v1/integration/status"
        try:
            status_response = requests.get(
                status_url,
                timeout=self.health_request_timeout,
            )
            status_response.raise_for_status()
            self.publish_recovery_log(
                "sonar_unavailable",
                "Sonar3d is available",
            )
        except requests.exceptions.RequestException as exc:
            self.get_logger().error(f"Sonar3d status request failed: {exc}")
            self.publish_log_once(
                "sonar_unavailable",
                LoggingSeverity.ERROR.value,
                "Sonar3d is not available",
            )
            return

        time_status_url = (
            f"http://{self.sonar_ip}/api/v1/integration/time/status"
        )
        try:
            time_status_response = requests.get(
                time_status_url,
                timeout=self.health_request_timeout,
            )
            time_status_response.raise_for_status()
            time_status = time_status_response.json()
            if (
                not isinstance(time_status, dict)
                or not isinstance(time_status.get("ntp_synced"), bool)
            ):
                raise ValueError("invalid time status response")

            if time_status["ntp_synced"]:
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
        except (requests.exceptions.RequestException, ValueError) as exc:
            self.get_logger().error(
                f"Sonar3d time status request failed: {exc}"
            )

        acoustics_url = (
            f"http://{self.sonar_ip}/api/v1/integration/acoustics/enabled"
        )
        try:
            acoustics_response = requests.get(
                acoustics_url,
                timeout=self.health_request_timeout,
            )
            acoustics_response.raise_for_status()
            acoustics_enabled = acoustics_response.json()
            if not isinstance(acoustics_enabled, bool):
                raise ValueError("expected a boolean response")
        except (requests.exceptions.RequestException, ValueError) as exc:
            self.get_logger().error(
                f"Sonar3d acoustics status request failed: {exc}"
            )
            return

        if self.desired_acoustics_enabled is None:
            self.desired_acoustics_enabled = acoustics_enabled

        if self.desired_acoustics_enabled and not acoustics_enabled:
            try:
                enable_response = self.set_acoustics(True)
                enable_response.raise_for_status()
                self.publish_recovery_log(
                    "acoustics_disabled",
                    "Sonar3d acoustics are enabled",
                )
                self.get_logger().info(
                    "Re-enabled Sonar3d acoustics after sonar restart"
                )
                return
            except requests.exceptions.RequestException as exc:
                self.get_logger().error(
                    f"Failed to re-enable Sonar3d acoustics: {exc}"
                )

        if not acoustics_enabled:
            self.publish_log_once(
                "acoustics_disabled",
                LoggingSeverity.WARN.value,
                "Sonar3d acoustics are not enabled",
            )
            return

        self.publish_recovery_log(
            "acoustics_disabled",
            "Sonar3d acoustics are enabled",
        )

    def start_sonar(self, req, res):
        self.get_logger().info("Start pinging")
        self.desired_acoustics_enabled = True
        try:
            api_res = self.set_acoustics(True)
            res.success = api_res.status_code == 204
            res.message = api_res.text
            if res.success:
                self.publish_recovery_log(
                    "acoustics_disabled",
                    "Sonar3d acoustics are enabled",
                )
        except requests.exceptions.RequestException as exc:
            self.get_logger().error(f"Failed to start Sonar3d acoustics: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def stop_sonar(self, req, res):
        self.get_logger().info("Stop pinging")
        self.desired_acoustics_enabled = False
        try:
            api_res = self.set_acoustics(False)
            res.success = api_res.status_code == 204
            res.message = api_res.text
        except requests.exceptions.RequestException as exc:
            self.get_logger().error(f"Failed to stop Sonar3d acoustics: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def get_status(self, req, res):
        try:
            api_res = requests.get(
                f"http://{self.sonar_ip}/api/v1/integration/status",
                timeout=self.health_request_timeout,
            )
            res.success = api_res.status_code == 200
            res.message = api_res.text
        except requests.exceptions.RequestException as exc:
            self.get_logger().error(f"Sonar3d status request failed: {exc}")
            res.success = False
            res.message = "Sonar3d is not available"
        return res

    def get_sonar_ip(self, req, res):
        res.success = True
        res.message = self.sonar_ip
        return res

    def parse_rip2_packet(self, data: bytes):
        """
        Parse the RIP2 framing:
          1. Verify the "RIP2" magic header
          2. Verify total_length field matches the data size
          3. Check CRC
          4. Extract payload (compressed protobuf data) from the packet
          5. Decompress the payload using Snappy

        Returns:
          payload (bytes) if valid, or None if there's an error.
        """
        if len(data) < 13:
            self.get_logger().warning(f"Packet too small: only {len(data)} bytes.")
            return None

        # First 4 bytes are "RIP2"
        magic = data[:4]
        if magic != b"RIP2":
            self.get_logger().warning(
                f"Invalid magic: got {magic!r} instead of b'RIP2'."
            )
            return None

        # Next 4 bytes (little-endian) specify the total packet length
        total_length = struct.unpack("<I", data[4:8])[0]
        if len(data) < total_length:
            self.get_logger().warning(
                f"Packet truncated: needed {total_length} bytes, got {len(data)}."
            )
            return None

        # The payload is between offset 8 and (total_length - 4)
        compressed_payload = data[8 : total_length - 4]

        # Last 4 bytes in the packet is the CRC32
        crc_received = struct.unpack("<I", data[total_length - 4 : total_length])[0]
        crc_calculated = zlib.crc32(data[: total_length - 4]) & 0xFFFFFFFF
        if crc_calculated != crc_received:
            self.get_logger().warning(
                f"CRC mismatch: expected 0x{crc_calculated:08x}, got 0x{crc_received:08x}."
            )
            return None

        # Decompress the payload
        try:
            payload = snappy.decompress(compressed_payload)
        except Exception as exc:
            self.get_logger().error(f"Snappy decompression error: {exc}")
            return None

        return payload

    def decode_protobuf_packet(self, payload: bytes):
        """
        Decode the Protobuf Packet (top-level), which may contain:
          - BitmapImageGreyscale8
          - RangeImage
          - or an unknown message type (google.protobuf.Any)

        Returns:
          (msg_type_name, message_object) if successfully parsed,
          or None if parsing failed.
        """
        # Create a top-level Packet object
        packet = Packet()
        try:
            packet.ParseFromString(payload)
        except Exception as exc:
            self.get_logger().error(f"Protobuf parse error: {exc}")
            return None

        # The actual data is in the .msg field (type google.protobuf.Any)
        any_msg = packet.msg
        if not any_msg.IsInitialized():
            return None

        # Attempt to unpack into BitmapImageGreyscale8
        bmp = BitmapImageGreyscale8()
        if any_msg.Unpack(bmp):
            return ("BitmapImageGreyscale8", bmp)

        # Otherwise, try to unpack into RangeImage
        rng = RangeImage()
        if any_msg.Unpack(rng):
            return ("RangeImage", rng)

        # If it's neither of the above, return Unknown
        return ("Unknown", any_msg)

    def to_np(self, img_obj, dtype):
        img = np.array(list(img_obj.image_pixel_data), dtype=dtype)
        img = img.reshape((img_obj.height, img_obj.width))
        return np.flip(img, 0)

    def pair2pc(self, range_img, int_img):
        range_img = np.flip(range_img, 1)
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
            data, address = self.sock.recvfrom(self.BUFFER_SIZE)
        except TimeoutError:
            return
        if address[0] != self.sonar_ip:
            return

        payload = self.parse_rip2_packet(data)
        if payload is None:
            return

        result = self.decode_protobuf_packet(payload)
        if result is None:
            return

        msg_type, msg_obj = result

        if msg_type == "Unknown":
            return

        self.HFOV = math.radians(msg_obj.fov_horizontal)
        self.VFOV = math.radians(msg_obj.fov_vertical)
        self.MAX_RANGE = msg_obj.range

        if msg_type == "BitmapImageGreyscale8":
            img = self.to_np(msg_obj, np.uint8)
            self.publish_img(
                img, img, self.int_pub, self.int_ui_pub, msg_obj.header.timestamp
            )

            if len(self.image_sync) == 0:
                return
            if self.image_sync[0] == msg_obj.header.sequence_id:
                self.process_img_pair(self.image_sync[1], img, msg_obj.header.timestamp)

        elif msg_type == "RangeImage":
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
