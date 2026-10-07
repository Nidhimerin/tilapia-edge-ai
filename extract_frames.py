import cv2, glob, os, sys

test_clips = set(sys.argv[1:])          # e.g. VID_20260908_144015234 VID-20260908-WA0065
every_s, max_per_clip = 3, 6

for v in sorted(glob.glob("footage/videos/*.mp4") + glob.glob("footage/videos/*.mov")):
    name = os.path.splitext(os.path.basename(v))[0]
    dest = "frames_test" if name in test_clips else "frames_train"
    os.makedirs(dest, exist_ok=True)
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * every_s)
    idxs = list(range(0, n, step))[:max_per_clip]
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(f"{dest}/{name}__{idx}.jpg", frame)
    cap.release()
print("done")