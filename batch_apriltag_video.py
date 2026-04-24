"""Batch AprilTag video + CSV for head (RealSense) or wrist cams from capture_log."""

import csv
import cv2
import numpy as np
import apriltag
from pathlib import Path

# --- Configure once per run ---
# Session folder name under the repo root (contains images/, capture_log.csv, etc.)
SESSION_DIR_NAME = "2026-04-24_10-49-21"
# "head" — images/head/*.jpg + realsense_log.csv timestamps
# "left" / "right" — capture_log.csv rows for the matching camera_id; filenames relative to SESSION
# "lucid" — session-root *.jpg + bayer_rg8_arrival.csv timestamps
STREAM = "lucid" # "head" or "left" or "right" or "lucid"
# Must match camera_id in capture_log.csv (e.g. left_video_0, right_video_1)
LEFT_CAMERA_ID = "left_video_0"
RIGHT_CAMERA_ID = "right_video_0"

REPO_ROOT = Path(__file__).resolve().parent
SESSION = REPO_ROOT / SESSION_DIR_NAME

HEAD_INPUT_DIR = SESSION / "images" / "head"
HEAD_PATTERN = "*.jpg"
LUCID_PATTERN = "frame_*.jpg"
HEAD_REALSENSE_CSV = SESSION / "realsense_log.csv"
LUCID_CAMERA_CSV = SESSION / "bayer_rg8_arrival.csv"
CAPTURE_LOG = SESSION / "capture_log.csv"

FPS = 30
TAG_FAMILY = "tagStandard41h12"
TARGET_IDS = {5, 6}

APRILTAG_THREADS = 4
APRILTAG_REFINE_EDGES = True
APRILTAG_DECIMATE = 1.0


def _make_apriltag_detector():
    kwargs = dict(
        threads=APRILTAG_THREADS,
        refine_edges=APRILTAG_REFINE_EDGES,
        decimate=APRILTAG_DECIMATE,
    )
    try:
        return apriltag.apriltag(TAG_FAMILY, **kwargs)
    except TypeError:
        return apriltag.apriltag(TAG_FAMILY)


def _detections_for_targets(detector, gray: np.ndarray) -> dict:
    return {d["id"]: d for d in detector.detect(gray) if d["id"] in TARGET_IDS}


def _merge_target_detections(detector, gray: np.ndarray) -> list:
    found = _detections_for_targets(detector, gray)
    if TARGET_IDS.issubset(found.keys()):
        return list(found.values())

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for d in detector.detect(clahe.apply(gray)):
        if d["id"] in TARGET_IDS and d["id"] not in found:
            found[d["id"]] = d
    if TARGET_IDS.issubset(found.keys()):
        return list(found.values())

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    for d in detector.detect(sharpened):
        if d["id"] in TARGET_IDS and d["id"] not in found:
            found[d["id"]] = d
    return list(found.values())


def detect_apriltag_and_annotate(frame: np.ndarray, detector) -> tuple:
    """Detect AprilTags 5 and 6, draw annotations, return (annotated_frame, unit_vector or None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = _merge_target_detections(detector, gray)

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


def _load_realsense_by_bare_name() -> dict:
    realsense_data = {}
    with open(str(HEAD_REALSENSE_CSV), newline="") as f:
        for row in csv.DictReader(f):
            bare_name = Path(row["filename"]).name
            realsense_data[bare_name] = (
                row["monotonic_ns"],
                row["sensor_ts_ms"],
                row["frame_time_ns"],
            )
    return realsense_data


def _load_capture_frames(camera_id: str) -> list:
    rows = []
    with open(str(CAPTURE_LOG), newline="") as f:
        for row in csv.DictReader(f):
            if row["camera_id"] != camera_id:
                continue
            rel = row["filename"].strip()
            rows.append(
                {
                    "path": SESSION / rel,
                    "monotonic_ns": row["monotonic_ns"].strip(),
                    "presentation_time_ns": row.get("presentation_time_ns", "").strip(),
                    "pts": row.get("pts", "").strip(),
                }
            )
    rows.sort(key=lambda r: int(r["monotonic_ns"]))
    return rows


def _load_lucid_frames() -> list:
    rows = []
    with open(str(LUCID_CAMERA_CSV), newline="") as f:
        for row in csv.DictReader(f):
            image_name = Path(row["jpeg"]).name
            rows.append(
                {
                    "path": SESSION / image_name,
                    "monotonic_ns": row["monotonic_ns"].strip(),
                }
            )
    rows.sort(key=lambda r: int(r["monotonic_ns"]))
    return rows


def run_head():
    output_mp4 = SESSION / "head_apriltag_processed.mp4"
    output_csv = SESSION / "head_apriltag_processed.csv"
    image_paths = sorted(HEAD_INPUT_DIR.glob(HEAD_PATTERN))
    if not image_paths:
        raise SystemExit(f"No images found matching {HEAD_INPUT_DIR / HEAD_PATTERN}")
    if not HEAD_REALSENSE_CSV.is_file():
        raise SystemExit(f"Missing realsense log: {HEAD_REALSENSE_CSV}")

    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        raise SystemExit(f"Failed to read sample image: {image_paths[0]}")

    realsense_data = _load_realsense_by_bare_name()
    frame_h, frame_w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (frame_w, frame_h))
    detector = _make_apriltag_detector()

    print(f"Writing video to {output_mp4} with {len(image_paths)} frames...")
    with open(output_csv, "w", newline="") as csv_file:
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
    print(f"Video generation complete: {output_mp4}")
    print(f"CSV generation complete: {output_csv}")


def run_lucid() -> None:
    output_mp4 = SESSION / "lucid_apriltag_processed.mp4"
    output_csv = SESSION / "lucid_apriltag_processed.csv"
    if not LUCID_CAMERA_CSV.is_file():
        raise SystemExit(f"Missing Lucid arrival log: {LUCID_CAMERA_CSV}")

    frames = _load_lucid_frames()
    if not frames:
        raise SystemExit(f"No Lucid frames listed in {LUCID_CAMERA_CSV}")

    sample = cv2.imread(str(frames[0]["path"]))
    if sample is None:
        raise SystemExit(f"Failed to read sample image: {frames[0]['path']}")

    frame_h, frame_w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (frame_w, frame_h))
    detector = _make_apriltag_detector()

    print(f"Writing video to {output_mp4} with {len(frames)} Lucid frames...")
    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["monotonic_ns", "vector_x", "vector_y"])

        for index, fr in enumerate(frames, start=1):
            image = cv2.imread(str(fr["path"]))
            if image is None:
                print(f"Warning: failed to read {fr['path']}, skipping")
                continue

            frame, unit_vector = detect_apriltag_and_annotate(image, detector)
            writer.write(frame)

            vx = f"{unit_vector[0]:.8f}" if unit_vector is not None else ""
            vy = f"{unit_vector[1]:.8f}" if unit_vector is not None else ""
            csv_writer.writerow([fr["monotonic_ns"], vx, vy])

            if index % 100 == 0 or index == len(frames):
                print(f"Processed {index}/{len(frames)} frames")

    writer.release()
    print(f"Video generation complete: {output_mp4}")
    print(f"CSV generation complete: {output_csv}")


def _run_capture_stream(side: str, camera_id: str) -> None:
    """side is 'left' or 'right' (used for output filenames only)."""
    output_mp4 = SESSION / f"{side}_apriltag_processed.mp4"
    output_csv = SESSION / f"{side}_apriltag_processed.csv"
    if not CAPTURE_LOG.is_file():
        raise SystemExit(f"Missing capture log: {CAPTURE_LOG}")

    frames = _load_capture_frames(camera_id)
    if not frames:
        raise SystemExit(f"No capture_log rows for camera_id={camera_id!r}")

    sample = cv2.imread(str(frames[0]["path"]))
    if sample is None:
        raise SystemExit(f"Failed to read sample image: {frames[0]['path']}")

    frame_h, frame_w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (frame_w, frame_h))
    detector = _make_apriltag_detector()

    print(f"Writing video to {output_mp4} with {len(frames)} frames (monotonic order)...")
    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["monotonic_ns", "presentation_time_ns", "pts", "vector_x", "vector_y"]
        )

        for index, fr in enumerate(frames, start=1):
            image = cv2.imread(str(fr["path"]))
            if image is None:
                print(f"Warning: failed to read {fr['path']}, skipping")
                continue

            frame, unit_vector = detect_apriltag_and_annotate(image, detector)
            writer.write(frame)

            vx = f"{unit_vector[0]:.8f}" if unit_vector is not None else ""
            vy = f"{unit_vector[1]:.8f}" if unit_vector is not None else ""
            csv_writer.writerow(
                [fr["monotonic_ns"], fr["presentation_time_ns"], fr["pts"], vx, vy]
            )

            if index % 100 == 0 or index == len(frames):
                print(f"Processed {index}/{len(frames)} frames")

    writer.release()
    print(f"Video generation complete: {output_mp4}")
    print(f"CSV generation complete: {output_csv}")


def run_left():
    _run_capture_stream("left", LEFT_CAMERA_ID)


def run_right():
    _run_capture_stream("right", RIGHT_CAMERA_ID)


if __name__ == "__main__":
    s = STREAM.strip().lower()
    if s == "head":
        run_head()
    elif s == "lucid":
        run_lucid()
    elif s == "left":
        run_left()
    elif s == "right":
        run_right()
    else:
        raise SystemExit(f"STREAM must be 'head', 'left', 'right', or 'lucid', got {STREAM!r}")
