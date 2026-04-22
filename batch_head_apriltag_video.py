import csv
import cv2
import numpy as np
import apriltag
from pathlib import Path

INPUT_DIR = Path("/home/roboforce/Desktop/latency_gen-umi/20260422_132234/images/head")
REALSENSE_CSV = Path("/home/roboforce/Desktop/latency_gen-umi/20260422_132234/realsense_log.csv")
PATTERN = "*.jpg"
OUTPUT_PATH = "/home/roboforce/Desktop/latency_gen-umi/head_apriltag_processed.mp4"
CSV_OUTPUT_PATH = "/home/roboforce/Desktop/latency_gen-umi/head_apriltag_processed.csv"
FPS = 30
TAG_FAMILY = "tagStandard41h12"
TARGET_IDS = {5, 6}


def detect_apriltag_and_annotate(frame: np.ndarray, detector) -> tuple:
    """Detect AprilTags 5 and 6, draw annotations, return (annotated_frame, unit_vector or None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    annotated = frame.copy()
    unit_vector = None
    found = {d["id"]: d for d in detections if d["id"] in TARGET_IDS}

    for det in found.values():
        corners = det["lb-rb-rt-lt"].astype(int)
        center = det["center"].astype(int)
        tag_id = det["id"]

        pts = corners.reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        corner_labels = ["LB", "RB", "RT", "LT"]
        for pt, label in zip(corners, corner_labels):
            cv2.circle(annotated, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(annotated, label, tuple(pt + [4, -4]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.circle(annotated, tuple(center), 6, (255, 0, 0), -1)
        cv2.putText(annotated, f"ID:{tag_id}", (center[0] - 20, center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if 5 in found and 6 in found:
        c5 = found[5]["center"]
        c6 = found[6]["center"]
        vec = c6 - c5
        dist = np.linalg.norm(vec)
        if dist > 0:
            unit_vector = vec / dist

        pt5 = tuple(c5.astype(int))
        pt6 = tuple(c6.astype(int))
        cv2.line(annotated, pt5, pt6, (255, 255, 0), 1, cv2.LINE_AA)

        if unit_vector is not None:
            arrow_scale = min(annotated.shape[:2]) * 0.15
            arrow_end = (c5 + unit_vector * arrow_scale).astype(int)
            cv2.arrowedLine(annotated, pt5, tuple(arrow_end),
                            color=(0, 165, 255), thickness=2,
                            line_type=cv2.LINE_AA, tipLength=0.25)

            mid = ((c5 + c6) / 2).astype(int)
            cv2.putText(annotated, f"uv=({unit_vector[0]:.3f}, {unit_vector[1]:.3f})",
                        (mid[0] + 5, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            # Top-left readout
            info_lines = [
                "Unit Vector (5->6):",
                f"  X: {unit_vector[0]:.4f}",
                f"  Y: {unit_vector[1]:.4f}",
            ]
            x0, y0, line_h = 10, 28, 26
            for i, text in enumerate(info_lines):
                y = y0 + i * line_h
                cv2.putText(annotated, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        detected_ids = sorted(found.keys())
        status = f"Tags found: {detected_ids}" if detected_ids else "No target tags found"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated, unit_vector


if __name__ == "__main__":
    image_paths = sorted(INPUT_DIR.glob(PATTERN))
    if not image_paths:
        raise SystemExit(f"No images found matching {INPUT_DIR / PATTERN}")

    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        raise SystemExit(f"Failed to read sample image: {image_paths[0]}")

    # Load realsense log — key by bare filename (e.g. "972381346313532.jpg")
    realsense_data = {}
    with open(str(REALSENSE_CSV), newline="") as f:
        for row in csv.DictReader(f):
            bare_name = Path(row["filename"]).name
            realsense_data[bare_name] = (
                row["monotonic_ns"],
                row["sensor_ts_ms"],
                row["frame_time_ns"],
            )

    frame_h, frame_w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (frame_w, frame_h))

    detector = apriltag.apriltag(TAG_FAMILY)

    print(f"Writing video to {OUTPUT_PATH} with {len(image_paths)} frames...")
    with open(CSV_OUTPUT_PATH, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["monotonic_ns", "sensor_ts_ms", "frame_time_ns", "vector_x", "vector_y"])

        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to read {image_path}, skipping")
                continue

            frame, unit_vector = detect_apriltag_and_annotate(image, detector)
            writer.write(frame)

            timestamps = realsense_data.get(image_path.name)
            if timestamps is not None:
                monotonic_ns, sensor_ts_ms, frame_time_ns = timestamps
                vx = f"{unit_vector[0]:.8f}" if unit_vector is not None else ""
                vy = f"{unit_vector[1]:.8f}" if unit_vector is not None else ""
                csv_writer.writerow([monotonic_ns, sensor_ts_ms, frame_time_ns, vx, vy])

            if index % 100 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)} frames")

    writer.release()
    print(f"Video generation complete: {OUTPUT_PATH}")
    print(f"CSV generation complete: {CSV_OUTPUT_PATH}")
