"""
RTSP Pipeline Solishtirish - 4 kamera
1) OpenCV FFmpeg (CPU decode) + grab thread
2) GStreamer NVDEC (GPU decode) + grab thread
3) GStreamer CPU (software decode) + grab thread
"""
import cv2
import time
import os
import threading
import subprocess

CAMS = [
    ("Shimoliy", "rtsp://admin:1q2w3e4r5t@192.168.170.160:554/Streaming/Channels/201"),
    ("Janubiy",  "rtsp://admin:1q2w3e4r5t@192.168.170.160:554/Streaming/Channels/301"),
    ("Sharqiy",  "rtsp://admin:1q2w3e4r5t@192.168.170.160:554/Streaming/Channels/801"),
    ("G'arbiy",  "rtsp://admin:1q2w3e4r5t@192.168.170.160:554/Streaming/Channels/601"),
]


def _run_grab_thread_test(caps, label, measure_cpu=False):
    """Umumiy grab thread test - har qanday backend uchun"""
    n = len(caps)
    latest = [[None] for _ in range(n)]
    locks = [threading.Lock() for _ in range(n)]
    running = [True]

    threads = []
    for i, (name, cap) in enumerate(caps):
        def grabber(c=cap, lat=latest[i], lk=locks[i]):
            while running[0]:
                ret = c.grab()
                if not ret:
                    time.sleep(0.01)
                    continue
                with lk:
                    need = lat[0] is None
                if need:
                    ret2, f = c.retrieve()
                    if ret2:
                        with lk:
                            lat[0] = f
        t = threading.Thread(target=grabber, daemon=True)
        t.start()
        threads.append(t)

    fps_c = [0]*n; fps_t = [time.time()]*n; fps_v = [0.0]*n
    decode_times = [[] for _ in range(n)]

    while True:
        for i, (name, _) in enumerate(caps):
            t0 = time.perf_counter()
            with locks[i]:
                frame = latest[i][0]
                latest[i][0] = None
            if frame is None:
                continue

            dt = (time.perf_counter() - t0) * 1000
            decode_times[i].append(dt)
            if len(decode_times[i]) > 100:
                decode_times[i].pop(0)

            fps_c[i] += 1
            now = time.time()
            if now - fps_t[i] >= 1.0:
                fps_v[i] = fps_c[i] / (now - fps_t[i])
                fps_c[i] = 0
                fps_t[i] = now

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

            info = f"{name} [{label}] FPS:{fps_v[i]:.1f}"
            cv2.putText(frame, info, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(f"{name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running[0] = False
    for t in threads:
        t.join(2)
    for _, cap in caps:
        cap.release()

    # Natijalar
    print(f"\n--- {label} Natijalar ---")
    for i, (name, _) in enumerate(caps):
        avg_fps = fps_v[i]
        print(f"  {name}: {avg_fps:.1f} FPS")


def test_opencv_ffmpeg():
    """Test 1: OpenCV + FFmpeg (CPU decode)"""
    print("\n=== TEST 1: OpenCV FFmpeg (CPU decode) ===")

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|stimeout;2000000|'
        'fflags;nobuffer+discardcorrupt|flags;low_delay|'
        'analyzeduration;100000|probesize;100000|'
        'max_delay;0|reorder_queue_size;0'
    )

    caps = []
    for name, url in CAMS:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  {name}: ochilmadi")
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Flush stale buffer
        for _ in range(30):
            cap.grab()
        caps.append((name, cap))
        print(f"  {name}: OK")

    if not caps:
        print("  Hech bir kamera ochilmadi!")
        return

    _run_grab_thread_test(caps, "FFmpeg CPU")


def test_gstreamer_nvdec():
    """Test 2: GStreamer + NVDEC (GPU decode) - H.265"""
    print("\n=== TEST 2: GStreamer NVDEC (GPU decode, H.265) ===")

    caps = []
    for name, url in CAMS:
        gst = (
            f"rtspsrc location={url} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! "
            f"nvh265dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"  {name}: ochilmadi (GStreamer NVDEC)")
            continue
        # Flush
        for _ in range(30):
            cap.grab()
        caps.append((name, cap))
        print(f"  {name}: OK (GPU decode)")

    if not caps:
        print("  Hech bir kamera ochilmadi! NVDEC plugin yo'q bo'lishi mumkin.")
        return

    _run_grab_thread_test(caps, "GStreamer NVDEC")


def test_gstreamer_cpu():
    """Test 3: GStreamer + avdec (CPU decode) - H.265"""
    print("\n=== TEST 3: GStreamer CPU (software decode, H.265) ===")

    caps = []
    for name, url in CAMS:
        gst = (
            f"rtspsrc location={url} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! "
            f"avdec_h265 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"  {name}: ochilmadi (GStreamer CPU)")
            continue
        for _ in range(30):
            cap.grab()
        caps.append((name, cap))
        print(f"  {name}: OK (CPU decode)")

    if not caps:
        print("  Hech bir kamera ochilmadi!")
        return

    _run_grab_thread_test(caps, "GStreamer CPU")


if __name__ == "__main__":
    print("=" * 50)
    print("  RTSP Pipeline Solishtirish (4 kamera)")
    print("=" * 50)

    # Tekshirish
    bi = cv2.getBuildInformation()
    has_gst = "GStreamer:                   YES" in bi
    has_ffmpeg = "FFMPEG:                      YES" in bi
    print(f"\n  OpenCV {cv2.__version__}")
    print(f"  FFmpeg:    {'✓' if has_ffmpeg else '✗'}")
    print(f"  GStreamer: {'✓' if has_gst else '✗'}")

    # NVDEC H.265 check
    try:
        r = subprocess.run(["gst-inspect-1.0", "nvh265dec"],
                           capture_output=True, text=True, timeout=3)
        has_nvdec = r.returncode == 0
    except Exception:
        has_nvdec = False
    print(f"  NVDEC H265:{'✓' if has_nvdec else '✗'}")
    print(f"  Kamera:    H.265 (HEVC)")

    print(f"\n  Kameralar: {len(CAMS)} ta")
    print()
    print("  1) OpenCV FFmpeg  (CPU decode) + grab thread")
    if has_gst and has_nvdec:
        print("  2) GStreamer NVDEC (GPU decode) + grab thread")
    else:
        print("  2) GStreamer NVDEC (mavjud emas)")
    if has_gst:
        print("  3) GStreamer CPU  (soft decode) + grab thread")
    else:
        print("  3) GStreamer CPU  (mavjud emas)")
    print()

    choice = input("Tanlang (1/2/3): ").strip()
    cv2.destroyAllWindows()

    if choice == "1":
        test_opencv_ffmpeg()
    elif choice == "2":
        test_gstreamer_nvdec()
    elif choice == "3":
        test_gstreamer_cpu()
    else:
        print("Noto'g'ri tanlov!")

    cv2.destroyAllWindows()
