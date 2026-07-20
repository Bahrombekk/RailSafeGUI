"""
Camera Helper - Optimized RTSP camera connection utilities
"""

import cv2
import os
import threading
import time
from typing import Optional, Tuple


class OptimizedCamera:
    """Optimized camera connection with fast timeout and reconnection"""

    def __init__(self, source: str, name: str = "Camera"):
        self.source = source
        self.name = name
        self.cap = None
        self.is_rtsp = source.startswith("rtsp://")
        self._setup_env()

    def _setup_env(self):
        """Setup environment variables for faster RTSP connection"""
        if self.is_rtsp:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|'
                'flags;low_delay|analyzeduration;1000000|probesize;500000'
            )
        # Suppress HEVC/codec warnings
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

    def open(self, timeout: float = 10.0) -> bool:
        """
        Open camera with timeout

        Args:
            timeout: Maximum time to wait for connection in seconds

        Returns:
            True if connected successfully, False otherwise
        """
        result = [False]
        cap_result = [None]
        lock = threading.Lock()
        cancelled = [False]

        def try_open():
            cap = None
            try:
                print(f"[INFO] Connecting to {self.name}...")
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

                if cap.isOpened():
                    # Optimize settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 25)

                    # Try to read first valid frame (skip corrupted frames)
                    ret = False
                    for _ in range(10):
                        ret, _frame = cap.read()
                        if ret:
                            break
                    if ret:
                        # Egalikni faqat asosiy thread hali kutayotgan bo'lsa beramiz.
                        # Timeout o'tib ketgan bo'lsa (cancelled) — cap'ni o'zimiz
                        # yopamiz, aks holda ochilgan RTSP/FFmpeg sessiyasi leak bo'lardi.
                        with lock:
                            if cancelled[0]:
                                cap.release()
                                return
                            cap_result[0] = cap
                            result[0] = True
                            cap = None  # egalik cap_result ga o'tdi
                        print(f"[OK] {self.name} connected successfully")
                    else:
                        print(f"[ERROR] Cannot read from {self.name}")
                        cap.release()
                        cap = None
                else:
                    print(f"[ERROR] Cannot open {self.name}")
            except Exception as e:
                print(f"[ERROR] Exception opening {self.name}: {e}")
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        # Run in thread with timeout
        thread = threading.Thread(target=try_open, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        with lock:
            if result[0]:
                self.cap = cap_result[0]
                return True
            # Timeout: thread hali ishlayotgan bo'lsa, keyin ochilgan cap'ni
            # o'zi release qilishi uchun belgilaymiz.
            cancelled[0] = True

        print(f"[TIMEOUT] Connection to {self.name} timed out after {timeout}s")
        return False

    def read(self) -> Tuple[bool, Optional[any]]:
        """Read a frame from the camera"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        try:
            return self.cap.read()
        except Exception:
            return False, None

    def release(self):
        """Release the camera"""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
