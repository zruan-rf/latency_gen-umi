#!/usr/bin/env python3
"""Detect tagStandard41h12 AprilTags (IDs 5 and 6) in an image."""

import sys
import argparse
import cv2
import numpy as np
import apriltag
import matplotlib.pyplot as plt

TARGET_IDS = {5, 6}
TAG_FAMILY = "tagStandard41h12"


def draw_detection(img, det):
    corners = det["lb-rb-rt-lt"].astype(int)  # shape (4, 2): lb, rb, rt, lt
    tag_id = det["id"]
    center = det["center"].astype(int)

    # Draw polygon outline
    pts = corners.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Mark each corner
    corner_labels = ["LB", "RB", "RT", "LT"]
    for i, (pt, label) in enumerate(zip(corners, corner_labels)):
        cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(img, label, tuple(pt + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Draw center
    cv2.circle(img, tuple(center), 6, (255, 0, 0), -1)

    # Label the tag ID
    label = f"ID:{tag_id}"
    cv2.putText(img, label, (center[0] - 20, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def detect_tags(image_path: str, target_ids: set = TARGET_IDS):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = apriltag.apriltag(TAG_FAMILY)
    detections = detector.detect(gray)

    if not detections:
        print("No AprilTags detected.")
        return img, []

    found = [d for d in detections if d["id"] in target_ids]

    print(f"Total detections: {len(detections)}  |  Target IDs {sorted(target_ids)} found: {len(found)}")
    print("-" * 60)

    for det in found:
        tag_id = det["id"]
        center = det["center"]
        hamming = det["hamming"]
        margin = det["margin"]
        corners = det["lb-rb-rt-lt"]

        print(f"Tag ID     : {tag_id}")
        print(f"  Center   : ({center[0]:.1f}, {center[1]:.1f})")
        print(f"  Corners  : lb={corners[0].tolist()}  rb={corners[1].tolist()}")
        print(f"             rt={corners[2].tolist()}  lt={corners[3].tolist()}")
        print(f"  Hamming  : {hamming}")
        print(f"  Margin   : {margin:.2f}")
        print()

        draw_detection(img, det)

    # Draw unit vector arrow from center of ID 5 to center of ID 6
    centers = {d["id"]: d["center"] for d in found}
    if 5 in centers and 6 in centers:
        c5 = centers[5]
        c6 = centers[6]
        vec = c6 - c5
        dist = np.linalg.norm(vec)
        unit_vec = vec / dist if dist > 0 else vec

        print(f"Vector 5→6  : ({vec[0]:.2f}, {vec[1]:.2f})")
        print(f"Unit vector : ({unit_vec[0]:.4f}, {unit_vec[1]:.4f})")
        print(f"Distance    : {dist:.2f} px")

        # Draw full line between centers
        pt5 = tuple(c5.astype(int))
        pt6 = tuple(c6.astype(int))
        cv2.line(img, pt5, pt6, (255, 255, 0), 1, cv2.LINE_AA)

        # Draw arrow representing unit vector, scaled for visibility
        arrow_scale = min(img.shape[:2]) * 0.15
        arrow_end = (c5 + unit_vec * arrow_scale).astype(int)
        cv2.arrowedLine(img, pt5, tuple(arrow_end),
                        color=(0, 165, 255), thickness=2,
                        line_type=cv2.LINE_AA, tipLength=0.25)

        # Annotate unit vector text near midpoint
        mid = ((c5 + c6) / 2).astype(int)
        uv_label = f"uv=({unit_vec[0]:.3f}, {unit_vec[1]:.3f})"
        cv2.putText(img, uv_label, (mid[0] + 5, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # Top-left corner: bold white unit vector readout
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            "Unit Vector (5->6):",
            f"  X: {unit_vec[0]:.4f}",
            f"  Y: {unit_vec[1]:.4f}",
        ]
        x0, y0, line_h = 10, 28, 26
        for i, text in enumerate(lines):
            y = y0 + i * line_h
            # Black outline for readability
            cv2.putText(img, text, (x0, y), font, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            # White bold text on top
            cv2.putText(img, text, (x0, y), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return img, found


def main():
    parser = argparse.ArgumentParser(
        description=f"Detect {TAG_FAMILY} AprilTags (IDs {sorted(TARGET_IDS)}) in an image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--save", metavar="OUT",
        help="Save annotated image to OUT (default: <image>_annotated.jpg)"
    )
    args = parser.parse_args()

    annotated, found = detect_tags(args.image, TARGET_IDS)

    # Determine output path
    if args.save:
        out_path = args.save
    else:
        base = args.image.rsplit(".", 1)[0]
        out_path = base + "_annotated.jpg"

    cv2.imwrite(out_path, annotated)
    print(f"Annotated image saved to: {out_path}")

    # Always show with matplotlib
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 7))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("AprilTag Detection — unit vector ID5 → ID6")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
