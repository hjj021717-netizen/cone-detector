"""
ROS 2 노드: 카메라 영상에서 라바콘을 검출해 /cones/detections 로 발행한다. (관측 전용)

- 위치 계산, TF, 카메라 실측값이 필요 없다. (지도 표시는 cone_mapper_node.py 에서 나중에)
- 입력은 둘 중 하나:  image_topic (sensor_msgs/Image)  또는  stream_url (HTTP MJPEG)
- 제어 토픽(/cmd_vel 등)은 발행하지 않는다.
이 노드는 ROS 환경에서 직접 실행해 본 코드가 아니므로 현장에서 동작을 확인할 것.
"""
import hashlib
import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

from mjpeg_client import MjpegClient


def file_sha256_12(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except OSError:
        return "unknown"


def image_msg_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy (cv_bridge 없이). 지원: bgr8, rgb8, mono8"""
    ch = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(msg.encoding)
    if ch is None:
        raise ValueError("지원하지 않는 인코딩: %s" % msg.encoding)
    arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if msg.encoding == "mono8":
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__("cone_detector")
        self.declare_parameter("image_topic", "")        # 예) /frontvideostream
        self.declare_parameter("stream_url", "")         # 예) http://<IP>:8090/stream
        self.declare_parameter("model_path", "cone_best.pt")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("rate_hz", 8.0)
        self.declare_parameter("max_frame_age", 1.0)     # 이보다 오래된 프레임은 stale 처리 [s]

        p = self.get_parameter
        topic, url = p("image_topic").value, p("stream_url").value
        if bool(topic) == bool(url):
            raise RuntimeError("image_topic 또는 stream_url 중 정확히 하나만 지정하세요.")
        self.conf, self.imgsz = p("conf").value, p("imgsz").value
        self.max_frame_age = p("max_frame_age").value

        self.model_path = p("model_path").value
        self.model = YOLO(self.model_path)
        self.model_id = file_sha256_12(self.model_path)

        self.latest = None            # (수신 시각, BGR 이미지) — ROS 입력용
        self.client = None
        self.last_t = None
        self.fps_t0, self.fps_n, self.fps, self.infer_ms = time.time(), 0, 0.0, 0.0
        self.detections, self.img_size = [], (0, 0)

        if topic:
            self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        else:
            self.client = MjpegClient(url)
            self.client.start()

        self.pub = self.create_publisher(String, "/cones/detections", 1)
        self.create_timer(1.0 / p("rate_hz").value, self.process)
        self.get_logger().info("ConeDetector 시작: source=%s model=%s(%s)" %
                               (topic or url, self.model_path, self.model_id))

    def on_image(self, msg):
        try:
            self.latest = (time.time(), image_msg_to_bgr(msg))
        except Exception as e:
            self.get_logger().warn("이미지 변환 실패: %s" % e, throttle_duration_sec=5.0)

    def next_frame(self):
        """아직 처리하지 않은 최신 프레임 (수신 시각, BGR) 또는 None"""
        if self.client is not None:
            item = self.client.latest()
            if item is None or item[0] == self.last_t:
                return None
            img = cv2.imdecode(np.frombuffer(item[1], np.uint8), cv2.IMREAD_COLOR)
            return (item[0], img) if img is not None else None
        item = self.latest
        return item if item is not None and item[0] != self.last_t else None

    def process(self):
        item = self.next_frame()
        if item is not None:
            self.last_t = item[0]
            self.detect(item[1])
        self.publish()

    def detect(self, img):
        h, w = img.shape[:2]
        t0 = time.time()
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        self.infer_ms = (time.time() - t0) * 1000.0
        self.img_size = (w, h)
        dets = []
        if len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, conf):
                dets.append({"x1": round(float(x1), 1), "y1": round(float(y1), 1),
                             "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                             "conf": round(float(c), 3),
                             "cx_norm": round(float((x1 + x2) / 2 / w), 3),     # 0(왼쪽)~1(오른쪽)
                             "bottom_norm": round(float(y2 / h), 3)})           # 0(위)~1(아래), 클수록 가까움
        self.detections = dets[:64]
        self.fps_n += 1
        if time.time() - self.fps_t0 >= 2.0:
            self.fps, self.fps_n, self.fps_t0 = self.fps_n / (time.time() - self.fps_t0), 0, time.time()

    def publish(self):
        age = (time.time() - self.last_t) if self.last_t else None
        connected = self.client.connected if self.client is not None else True
        state = {
            "stamp": time.time(),
            "model": {"path": self.model_path, "sha256_12": self.model_id},
            "image": {"width": self.img_size[0], "height": self.img_size[1]},
            "health": {"source_connected": connected, "frame_age_s": age,
                       "stale": age is None or age > self.max_frame_age,
                       "fps": round(self.fps, 1), "infer_ms": round(self.infer_ms, 1)},
            "count": len(self.detections),
            "detections": [] if (age is None or age > self.max_frame_age) else self.detections,
        }
        self.pub.publish(String(data=json.dumps(state, separators=(",", ":"))))

    def destroy_node(self):
        if self.client is not None:
            self.client.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
