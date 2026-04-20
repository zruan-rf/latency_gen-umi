import csv
import cv2 as cv
import numpy as np
from pathlib import Path

INPUT_DIR = Path("/home/roboforce/Desktop/latency_gen-umi/20260419_142706/captures")
PATTERN = "*.jpg"
OUTPUT_PATH = "/home/roboforce/Desktop/latency_gen-umi/realsense_processed.mp4"
CSV_OUTPUT_PATH = "/home/roboforce/Desktop/latency_gen-umi/realsense_processed.csv"
FPS = 20
CROP_TOP = 50
CROP_BOTTOM = 450
CROP_LEFT = 80
CROP_RIGHT = 400


def detect_line_and_annotate(frame: np.ndarray) -> np.ndarray:
    cropped = frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    edges = cv.Canny(gray, 20, 80, apertureSize=3)

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=90, minLineLength=100, maxLineGap=10)

    line_img = cropped.copy()
    vector_text = "No line found"
    unit_vector = None
    line_endpoints = None

    if lines is not None and len(lines) > 0:
        best = max(lines[:, 0, :], key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1]))
        x1, y1, x2, y2 = best
        line_endpoints = (x1, y1, x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length > 0:
            unit_vector = (dx / length, dy / length)
            # Canonicalize: always point upward (uy > 0, or uy==0 and ux >= 0)
            if unit_vector[1] < 0 or (unit_vector[1] == 0.0 and unit_vector[0] < 0):
                unit_vector = (-unit_vector[0], -unit_vector[1])
                line_endpoints = (x2, y2, x1, y1)
            vector_text = f"Unit vector = ({unit_vector[0]:.5f}, {unit_vector[1]:.5f})"

    if unit_vector is None:
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv.contourArea)
            if cv.contourArea(biggest) > 50:
                vx, vy, x0, y0 = cv.fitLine(biggest, cv.DIST_L2, 0, 0.01, 0.01)
                if vx != 0 or vy != 0:
                    unit_vector = (float(vx), float(vy))
                    norm = np.hypot(unit_vector[0], unit_vector[1])
                    unit_vector = (unit_vector[0] / norm, unit_vector[1] / norm)
                    # Canonicalize: always point upward (uy > 0, or uy==0 and ux >= 0)
                    if unit_vector[1] < 0 or (unit_vector[1] == 0.0 and unit_vector[0] < 0):
                        unit_vector = (-unit_vector[0], -unit_vector[1])
                    vector_text = f"Unit vector = ({unit_vector[0]:.5f}, {unit_vector[1]:.5f})"
                    cx, cy = cropped.shape[1] / 2, cropped.shape[0] / 2
                    length = min(cropped.shape[1], cropped.shape[0]) * 0.45
                    x1 = int(cx - unit_vector[0] * length)
                    y1 = int(cy - unit_vector[1] * length)
                    x2 = int(cx + unit_vector[0] * length)
                    y2 = int(cy + unit_vector[1] * length)
                    line_endpoints = (x1, y1, x2, y2)

    if unit_vector is not None and line_endpoints is not None:
        x1, y1, x2, y2 = line_endpoints
        cv.line(line_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
        arrow_tip = (int(x1 + unit_vector[0] * 80), int(y1 + unit_vector[1] * 80))
        cv.arrowedLine(line_img, (x1, y1), arrow_tip, color=(0, 255, 0), thickness=3, tipLength=0.15)
        cv.circle(line_img, (x1, y1), radius=6, color=(0, 255, 0), thickness=-1)
    cv.putText(line_img, vector_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
    
    return line_img, unit_vector


def annotate_filename(frame: np.ndarray, filename: str) -> np.ndarray:
    cv.putText(frame, filename, (20, frame.shape[0] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
    return frame


if __name__ == "__main__":
    image_paths = sorted(INPUT_DIR.glob(PATTERN))
    if not image_paths:
        raise SystemExit(f"No images found matching {INPUT_DIR / PATTERN}")

    sample = cv.imread(str(image_paths[0]))
    if sample is None:
        raise SystemExit(f"Failed to read sample image {image_paths[0]}")

    # Load capture log for cam4 timestamps
    capture_data = {}
    with open(str(INPUT_DIR / "realsense_log.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["camera_name"] == "head":
                capture_data[Path(row["filename"]).name] = (
                    row["monotonic_ns"],
                    row["frame_time_ns"],
                )

    frame_height = CROP_BOTTOM - CROP_TOP
    frame_width = CROP_RIGHT - CROP_LEFT
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (frame_width, frame_height))

    print(f"Writing video to {OUTPUT_PATH} with {len(image_paths)} frames...")
    with open(CSV_OUTPUT_PATH, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["monotonic_ns", "frame_time_ns", "vector_x", "vector_y"])

        for index, image_path in enumerate(image_paths, start=1):
            image = cv.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to read {image_path}, skipping")
                continue

            frame, unit_vector = detect_line_and_annotate(image)
            annotate_filename(frame, image_path.name)
            writer.write(frame)

            timestamps = capture_data.get(image_path.name)
            if timestamps is not None:
                monotonic_ns, frame_time_ns = timestamps
                vx = f"{unit_vector[0]:.8f}" if unit_vector is not None else ""
                vy = f"{unit_vector[1]:.8f}" if unit_vector is not None else ""
                csv_writer.writerow([monotonic_ns, frame_time_ns, vx, vy])

            if index % 100 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)} frames")

    writer.release()
    print(f"Video generation complete: {OUTPUT_PATH}")
    print(f"CSV generation complete: {CSV_OUTPUT_PATH}")
