
import cv2
from pathlib import Path

base = Path("out_feeding_stationary")
frames_dir = base / "frames"
labels_dir = base / "labels"
output_dir = base / "boxed_frames"
output_dir.mkdir(exist_ok=True)

for frame_path in sorted(frames_dir.glob("*.jpg")):
    image = cv2.imread(str(frame_path))

    if image is None:
        continue

    height, width = image.shape[:2]
    label_path = labels_dir / f"{frame_path.stem}.txt"

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            class_id, x, y, box_w, box_h = map(float, values)

            x1 = int((x - box_w / 2) * width)
            y1 = int((y - box_h / 2) * height)
            x2 = int((x + box_w / 2) * width)
            y2 = int((y + box_h / 2) * height)

            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                image, "fish", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    cv2.imwrite(str(output_dir / frame_path.name), image)

print(f"Saved visualizations to: {output_dir}")
