"""Quick probe: what fps does the camera deliver at each resolution?"""
import cv2, time

CONFIGS = [
    (1280, 720,  30),
    (1280, 720,  60),
    (1280, 800,  30),
    (800,  600,  60),
    (640,  480,  60),
    (640,  480, 100),
    (640,  480, 120),
    (640,  360, 120),
    (320,  240, 120),
    (320,  240, 180),
]

for w, h, req_fps in CONFIGS:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)
    time.sleep(0.3)
    for _ in range(8):
        cap.read()
    t0 = time.time()
    n = 0
    while time.time() - t0 < 1.5:
        ok, _ = cap.read()
        if ok:
            n += 1
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"requested {w}x{h} @ {req_fps:>3}: "
          f"got {aw}x{ah}, measured {n/1.5:5.1f} fps")
    cap.release()
