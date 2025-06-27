#!/usr/bin/env python3

import socket
import struct
import zlib
import numpy as np
import rclpy
import requests
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
from std_srvs.srv import Trigger
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2 as cv
import math

from sonar3d_driver.sonar_3d_15_protocol_pb2 import (
    Packet,
    BitmapImageGreyscale8,
    RangeImage
)

class Sonar3d_driver(Node):
    BUFFER_SIZE = 65535
    MULTICAST_GROUP = "224.0.0.96"
    SONAR_IP = "192.168.2.199"
    PORT = 4747

    VFOV = math.radians(40)
    HFOV = math.radians(90)
    MAX_RANGE = 15

    def __init__(self):
        super().__init__('sonar3d_driver')
        
        self.range_pub = self.create_publisher(CompressedImage, 'sonar3d/range/compressed', 1)
        self.range_ui_pub = self.create_publisher(CompressedImage, 'sonar3d/range/ui/compressed', 1)
        self.int_pub = self.create_publisher(CompressedImage, 'sonar3d/intensity/compressed', 1)
        self.int_ui_pub = self.create_publisher(CompressedImage, 'sonar3d/intensity/ui/compressed', 1)
        self.range_int_pub = self.create_publisher(CompressedImage, 'sonar3d/range_intensity/compressed', 1)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'sonar3d/pointcloud', 1)

        self.start_srv = self.create_service(Trigger, 'sonar3d/start', self.start_sonar)
        self.stop_srv = self.create_service(Trigger, 'sonar3d/stop', self.stop_sonar)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.01, self.loop)

        self.MY_IP = socket.gethostbyname(socket.gethostname())

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.PORT))
        self.sock.settimeout(0.2)

        group = socket.inet_aton(self.MULTICAST_GROUP)
        mreq = struct.pack('4s4s', group, socket.inet_aton(self.MY_IP))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        self.image_sync = []

        print(f"Sonar3d driver started")

    def set_acoustics(self, value):
        res = requests.post(f"http://{self.SONAR_IP}/api/v1/integration/acoustics/enabled", json=value)
        return res.status_code == 204

    def start_sonar(self, req, res):
        print("Start pinging")
        res.success = self.set_acoustics(True)
        return res

    def stop_sonar(self, req, res):
        print("Stop pinging")
        res.success = self.set_acoustics(False)
        return res

    def parse_rip1_packet(self, data: bytes):
        """
        Parse the RIP1 framing:
          1. Verify the "RIP1" magic header
          2. Verify total_length field matches the data size
          3. Check CRC
          4. Extract payload (proto data) from the packet

        Returns:
          payload (bytes) if valid, or None if there's an error.
        """
        if len(data) < 13:
            print(f"Packet too small: only {len(data)} bytes.")
            return None

        # First 4 bytes are "RIP1"
        magic = data[:4]
        if magic != b'RIP1':
            print(f"Invalid magic: got {magic!r} instead of b'RIP1'.")
            return None

        # Next 4 bytes (little-endian) specify the total packet length
        total_length = struct.unpack('<I', data[4:8])[0]
        if len(data) < total_length:
            print(
                f"Packet truncated: needed {total_length} bytes, got {len(data)}.")
            return None

        # The payload is between offset 8 and (total_length - 4)
        payload = data[8: total_length - 4]

        # Last 4 bytes in the packet is the CRC32
        crc_received = struct.unpack('<I', data[total_length - 4: total_length])[0]
        crc_calculated = zlib.crc32(data[: total_length - 4]) & 0xffffffff
        if crc_calculated != crc_received:
            print(
                f"CRC mismatch: expected 0x{crc_calculated:08x}, got 0x{crc_received:08x}.")
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
        except Exception as e:
            print(f"Protobuf parse error: {e}")
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
        for i in range(range_img.shape[0]):
            for j in range(range_img.shape[1]):
                radius = range_img[i,j]
                if radius == 0: continue
                yaw = (j / (range_img.shape[1] - 1))  * self.HFOV - (self.HFOV / 2)
                pitch = (i / (range_img.shape[0] - 1)) * self.VFOV - (self.VFOV / 2)
                x = radius * math.cos(pitch) * math.cos(yaw);
                y = radius * math.cos(pitch) * math.sin(yaw);
                z = -radius * math.sin(pitch);
                pc.append((x,y,z,int_img[i,j]))

        header = self.get_header()

        if len(pc) == 0: 
            msg = PointCloud2()
            msg.header = header
            return msg

        pc = np.array(pc)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
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
        if address[0] != self.SONAR_IP: return
        
        payload = self.parse_rip1_packet(data)
        if payload is None: return

        result = self.decode_protobuf_packet(payload)
        if result is None: return

        msg_type, msg_obj = result
        if msg_type == "BitmapImageGreyscale8":
            img = self.to_np(msg_obj, np.uint8)
            self.publish_img(img, img, self.int_pub, self.int_ui_pub)

            if len(self.image_sync) == 0: return
            if self.image_sync[0] == msg_obj.header.sequence_id: self.process_img_pair(self.image_sync[1], img)

        elif msg_type == "RangeImage":
            img = self.to_np(msg_obj, np.uint32).astype(float)
            img *= msg_obj.image_pixel_scale 
            self.publish_img(img, (255*img/self.MAX_RANGE).astype(np.uint8), self.range_pub, self.range_ui_pub)

            # Range message are received first
            self.image_sync = [msg_obj.header.sequence_id, img]

    def process_img_pair(self, range_img, int_img):
        # print("Sonar3d image")

        pc = self.pair2pc(range_img, int_img)
        self.pointcloud_pub.publish(pc)

        new_shape = list(range_img.shape)
        new_shape.append(1)

        range_int_img = np.concatenate((range_img.reshape(new_shape), int_img.reshape(new_shape), np.zeros(new_shape)), axis=2)
        msg = self.bridge.cv2_to_compressed_imgmsg(range_int_img)
        msg.header = pc.header

        self.range_int_pub.publish(msg)

    def publish_img(self, img, img_ui, pub, pub_ui):
        msg = self.bridge.cv2_to_compressed_imgmsg(img)
        msg.header = self.get_header()

        img_ui = cv.applyColorMap(img_ui, cv.COLORMAP_JET)
        msg_ui = self.bridge.cv2_to_compressed_imgmsg(img_ui)
        msg_ui.header = msg.header

        pub.publish(msg)
        pub_ui.publish(msg_ui)

    def __del__(self):
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

if __name__ == '__main__':
    main()

