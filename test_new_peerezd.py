import numpy as np
import json
import yaml
import time
import threading
from queue import Queue
import torch
from pathlib import Path
from datetime import datetime
import traceback
import cv2
from ultralytics import YOLO


class SimpleVideoCapture:
    """cv2.VideoCapture bilan oddiy va barqaror RTSP/video o'qish"""

    def __init__(self, source, width=1920, height=1080):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            raise RuntimeError("VideoCapture failed")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[INFO] Opened: {self.source}")

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            print("[WARNING] Reopening capture...")
            self._open()
            time.sleep(0.3)
        ret, frame = self.cap.read()
        if not ret:
            print("[WARNING] Frame read failed → reconnecting")
            self.cap.release()
            self._open()
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()


class ImageSaver:
    def __init__(self, save_dir='saved_images'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.queue = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                if not self.queue.empty():
                    data = self.queue.get(timeout=1)
                    self._save_image(data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] ImageSaver: {e}")

    def _save_image(self, data):
        frame = data['frame']
        camera_id = data['camera_id']
        track_id = data['track_id']
        event_type = data['event_type']
        timestamp = data['timestamp']
        box_coords = data['box_coords']
        time_in_polygon = data.get('time_in_polygon', 0)
        class_id = data.get('class_id', 0)

        img = frame.copy()
        x1, y1, x2, y2 = box_coords

        colors = {'enter': (0, 255, 0), 'exit': (255, 0, 0), 'violation': (0, 0, 255)}
        color = colors.get(event_type, (255, 255, 255))
        event_text = event_type.upper()

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, event_text, (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if time_in_polygon > 0:
            cv2.putText(img, f"Time: {time_in_polygon:.1f}s", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"cam{camera_id}_{event_type}_id{track_id}_{ts}.jpg"

        cam_dir = self.save_dir / f"camera_{camera_id}"
        cam_dir.mkdir(exist_ok=True)
        event_dir = cam_dir / event_type
        event_dir.mkdir(exist_ok=True)
        filepath = event_dir / filename

        cv2.imwrite(str(filepath), img)

        h, w = frame.shape[:2]
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        with open(filepath.with_suffix('.txt'), 'w') as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def add_to_queue(self, **kwargs):
        kwargs['timestamp'] = datetime.now()
        self.queue.put(kwargs)

    def stop(self):
        self.running = False
        self.thread.join()


class DisplayThread:
    def __init__(self):
        self.queue = Queue(maxsize=5)
        self.running = True
        self.windows = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                if not self.queue.empty():
                    data = self.queue.get(timeout=0.5)
                    frame = data['frame']
                    win = data['window']

                    if win not in self.windows:
                        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                        h, w = frame.shape[:2]
                        cv2.resizeWindow(win, w, h)
                        self.windows[win] = True

                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                else:
                    time.sleep(0.01)
            except:
                pass

    def add(self, frame, window_name):
        if not self.running:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.queue.put({'frame': frame, 'window': window_name})

    def stop(self):
        self.running = False
        self.thread.join()
        cv2.destroyAllWindows()


class PolygonCamera:
    def __init__(self, cam_config, model_path, model_cfg, thresholds, proc_cfg, saver, display=None):
        self.id = cam_config['id']
        self.name = cam_config['name']
        self.source = cam_config['source']
        self.polygon_file = cam_config['polygon_file']
        self.saver = saver
        self.display = display

        print(f"[INFO] Loading YOLO model for cam {self.id}...")
        self.model = YOLO(model_path)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            self.model.fuse()

        self.target_classes = model_cfg['target_classes']
        self.threshold_warning = thresholds['warning']
        self.threshold_violation = thresholds['violation']

        self.adaptive = proc_cfg.get('adaptive_mode', True)
        self.skip_idle = proc_cfg.get('frame_skip_idle', 3)
        self.skip_active = proc_cfg.get('frame_skip_active', 1)

        self.cap = SimpleVideoCapture(self.source)

        # Polygon loading
        with open(self.polygon_file, 'r') as f:
            poly_data = json.load(f)

        orig_w = poly_data['images'][0]['width']
        orig_h = poly_data['images'][0]['height']
        scale_x = 1920 / orig_w
        scale_y = 1080 / orig_h

        pts = np.array(poly_data['annotations'][0]['segmentation'][0]).reshape(-1, 2)
        self.poly_pts = (pts * [scale_x, scale_y]).astype(np.int32)

        self.poly_mask = np.zeros((1080, 1920), np.uint8)
        cv2.fillPoly(self.poly_mask, [self.poly_pts], 255)

        self.fps_start = time.time()
        self.fps_count = 0
        self.fps = 0
        self.frame_cnt = 0
        self.proc_cnt = 0
        self.running = True
        self.skip = self.skip_idle
        self.frame_counter = 0
        self.empty_streak = 0
        self.empty_thresh = 3

        # 🔥 TRACKING DATA
        self.tracks = {}

        # 🚀 YANGI - SANASH TIZIMI
        self.counted_ids = set()  # ✅ Allaqachon o'tgan ID lar
        self.light_count = 0  # ✅ Yengil mashina (car)
        self.heavy_count = 0  # ✅ Og'ir mashina (bus, truck)
        self.total_count = 0  # ✅ Jami

        self.current_time = 0
        self.timeout = 3.0
        self.state = "empty"
        self.max_time = 0
        self.last_results = None

        self.color_safe = (255, 0, 0)
        self.color_warn = (0, 255, 255)
        self.color_viol = (0, 0, 255)
        self.color_out = (0, 255, 0)
        self.color_empty = (0, 255, 0)
        self.color_detected = (0, 255, 255)

        self.win = f"Cam {self.id} - {self.name}"

    def _update_fps(self):
        self.fps_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1:
            self.fps = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.time()

    def _in_poly(self, x, y):
        x, y = int(x), int(y)
        if 0 <= y < 1080 and 0 <= x < 1920:
            return self.poly_mask[y, x] > 0
        return False

    def _update_state(self):
        inside = 0
        max_t = 0
        for tr in self.tracks.values():
            if tr['in']:
                inside += 1
                max_t = max(max_t, tr['time'])

        if inside == 0:
            self.max_time = 0
            self.state = "empty"
        elif max_t >= self.threshold_violation:
            self.max_time = max_t
            self.state = "violation"
        else:
            self.max_time = max_t
            self.state = "detected"

    def _draw_poly(self, frame):
        color = self.color_empty if self.state == "empty" else \
            self.color_detected if self.state == "detected" else self.color_viol

        text = "BO'SH" if self.state == "empty" else \
            f"MAVJUD ({self.max_time:.1f}s)" if self.state == "detected" else \
                f"BUZILISH ({self.max_time:.1f}s)"

        cv2.polylines(frame, [self.poly_pts], True, color, 3)
        cv2.putText(frame, text, (10, 1050),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _get_color(self, t_in):
        if t_in < self.threshold_warning:
            return self.color_safe
        if t_in < self.threshold_violation:
            return self.color_warn
        return self.color_viol

    def run(self):
        print(f"[START] Camera {self.id} - {self.name}")

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.2)
                    continue

                self.frame_cnt += 1
                self.current_time = time.time()
                self.frame_counter += 1
                self._update_fps()

                process = self.frame_counter % self.skip == 0

                if process:
                    self.proc_cnt += 1

                    # 🔥 YOLO TRACKING
                    results = self.model.track(
                        frame,
                        classes=self.target_classes,
                        conf=0.45,
                        iou=0.5,
                        device=0 if torch.cuda.is_available() else "cpu",
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                        half=torch.cuda.is_available()
                    )

                    self.last_results = results[0]
                    dets = len(results[0].boxes) if results[0].boxes is not None else 0

                    if self.adaptive:
                        if dets == 0:
                            self.empty_streak += 1
                            if self.empty_streak >= self.empty_thresh:
                                self.skip = self.skip_idle
                        else:
                            self.empty_streak = 0
                            self.skip = self.skip_active

                    # Eski tracklarni tozalash
                    expired = [tid for tid, d in self.tracks.items()
                               if self.current_time - d['last'] > self.timeout]
                    for tid in expired:
                        del self.tracks[tid]

                    # 🔥 TRACKING NATIJALARINI ISHLASH
                    if results[0].boxes is not None and len(results[0].boxes):
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)

                        if results[0].boxes.id is not None:
                            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = np.arange(len(boxes))

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[i]
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            cls = clss[i]
                            track_id = track_ids[i]

                            inside = self._in_poly(cx, cy)

                            # Yangi track yaratish
                            if track_id not in self.tracks:
                                self.tracks[track_id] = {
                                    'cls': cls,
                                    'start': None,
                                    'in': False,
                                    'time': 0.0,
                                    'last': self.current_time,
                                    'viol_saved': False,
                                    'exit_saved': False
                                }

                            tr = self.tracks[track_id]

                            if inside:
                                if not tr['in']:
                                    # 🚀 POLYGON ICHIGA KIRDI - SANASH
                                    if track_id not in self.counted_ids:
                                        self.counted_ids.add(track_id)
                                        self.total_count += 1

                                        # ✅ CLASS BO'YICHA SANASH
                                        if cls == 2:  # car
                                            self.light_count += 1
                                            print(f"[COUNT] Yengil mashina #{self.light_count} (ID: {track_id})")
                                        elif cls in [5, 7]:  # bus, truck
                                            self.heavy_count += 1
                                            print(f"[COUNT] Og'ir mashina #{self.heavy_count} (ID: {track_id})")

                                    tr['start'] = self.current_time
                                    tr['in'] = True
                                    tr['viol_saved'] = False
                                    tr['exit_saved'] = False

                                    self.saver.add_to_queue(
                                        frame=frame,
                                        camera_id=self.id,
                                        track_id=track_id,
                                        event_type='enter',
                                        box_coords=(x1, y1, x2, y2),
                                        class_id=cls
                                    )

                                tr['time'] = self.current_time - tr['start']

                                if tr['time'] >= self.threshold_violation and not tr['viol_saved']:
                                    self.saver.add_to_queue(
                                        frame=frame,
                                        camera_id=self.id,
                                        track_id=track_id,
                                        event_type='violation',
                                        box_coords=(x1, y1, x2, y2),
                                        time_in_polygon=tr['time'],
                                        class_id=cls
                                    )
                                    tr['viol_saved'] = True
                            else:
                                if tr['in'] and not tr['exit_saved']:
                                    self.saver.add_to_queue(
                                        frame=frame,
                                        camera_id=self.id,
                                        track_id=track_id,
                                        event_type='exit',
                                        box_coords=(x1, y1, x2, y2),
                                        time_in_polygon=tr['time'],
                                        class_id=cls
                                    )
                                    tr['in'] = False
                                    tr['exit_saved'] = True

                            tr['last'] = self.current_time

                            # Vizualizatsiya
                            col = self._get_color(tr['time']) if tr['in'] else self.color_out
                            t_txt = f"{tr['time']:.1f}s" if tr['in'] else "Tashqarida"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                            cv2.putText(frame, f"ID{track_id} {t_txt}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                    self._update_state()

                # Frame chizish
                self._draw_poly(frame)
                cv2.putText(frame, f"{self.name} | FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (227, 30, 206), 2)

                inside_cnt = sum(1 for v in self.tracks.values() if v['in'])
                cv2.putText(frame, f"Inside: {inside_cnt}", (10, 1020),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 🚀 SANASH NATIJALARINI KO'RSATISH
                y_pos = 60
                cv2.putText(frame, f"Yengil: {self.light_count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
                cv2.putText(frame, f"Og'ir: {self.heavy_count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_pos += 30
                cv2.putText(frame, f"Jami: {self.total_count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if self.display:
                    self.display.add(frame, self.win)

            except Exception as e:
                print(f"[ERROR cam {self.id}] {e}")
                traceback.print_exc()
                time.sleep(0.5)

        self.cap.release()
        print(f"[STOP] Cam {self.id}")

    def stop(self):
        self.running = False


class MultiCameraSystem:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.saver = ImageSaver()
        self.display = DisplayThread()

        self.model_path = cfg['model']['path']
        self.model_cfg = {'target_classes': cfg['model']['target_classes']}
        self.thresh = cfg['thresholds']
        self.proc = cfg['processing']

        self.cameras = []
        for c in cfg.get('cameras', []):
            if c.get('enabled', False):
                cam = PolygonCamera(c, self.model_path, self.model_cfg,
                                    self.thresh, self.proc, self.saver, self.display)
                self.cameras.append(cam)

    def start(self):
        if not self.cameras:
            print("[ERROR] No enabled cameras")
            return

        threads = []
        for cam in self.cameras:
            t = threading.Thread(target=cam.run, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.2)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n[STOPPING]")
            for cam in self.cameras:
                cam.stop()
                print(f"\n{'=' * 50}")
                print(f"Kamera {cam.id} - {cam.name} YAKUNIY NATIJALAR:")
                print(f"{'=' * 50}")
                print(f"🚗 Yengil mashinalar: {cam.light_count}")
                print(f"🚚 Og'ir mashinalar: {cam.heavy_count}")
                print(f"📊 JAMI: {cam.total_count}")
                print(f"{'=' * 50}\n")
            self.saver.stop()
            self.display.stop()


if __name__ == "__main__":
    try:
        sys = MultiCameraSystem('config.yaml')
        sys.start()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()