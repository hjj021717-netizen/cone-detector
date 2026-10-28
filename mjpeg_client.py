"""HTTP MJPEG 스트림에서 최신 JPEG 한 장만 유지하는 클라이언트 (표준 라이브러리만 사용)."""
import threading
import time
import urllib.request


class MjpegClient(threading.Thread):
    def __init__(self, url, timeout=5.0, max_buffer=4 * 1024 * 1024):
        super().__init__(daemon=True)
        self.url = url
        self.timeout = timeout
        self.max_buffer = max_buffer
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # (수신 시각, jpeg bytes)
        self.frames = 0
        self.last_error = ""
        self.connected = False

    def stop(self):
        self._stop_evt.set()

    def latest(self):
        with self._lock:
            return self._latest

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                    self.connected, self.last_error = True, ""
                    buf = b""
                    while not self._stop_evt.is_set():
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            a = buf.find(b"\xff\xd8")
                            if a < 0:
                                buf = buf[-1:]
                                break
                            b = buf.find(b"\xff\xd9", a + 2)
                            if b < 0:
                                buf = buf[a:]
                                break
                            jpg, buf = buf[a:b + 2], buf[b + 2:]
                            with self._lock:
                                self._latest = (time.time(), jpg)
                            self.frames += 1
                        if len(buf) > self.max_buffer:      # 비정상 스트림 보호
                            buf = b""
            except Exception as e:              # 연결 실패/끊김: 잠시 후 재연결
                self.last_error = str(e)
            self.connected = False
            self._stop_evt.wait(1.0)
