import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = "/home/roboforce/Desktop/latency_gen-umi/20260419_142706/captures/790541308155782.jpg"
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

CROP_TOP = 50
CROP_BOTTOM = 450
CROP_LEFT = 0
CROP_RIGHT = 400

cropped = img[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]

# Convert to grayscale and detect edges for line extraction.
gray = cv.cvtColor(cropped, cv.COLOR_RGB2GRAY)

# Enhance contrast before edge detection (helps with faint lines on white paper).
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
gray = cv.GaussianBlur(gray, (3, 3), 0)

edges = cv.Canny(gray, 20, 80, apertureSize=3)

# Try probabilistic Hough transform first.
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=90, minLineLength=100, maxLineGap=10)

line_img = cropped.copy()
vector_text = "No line found"
unit_vector = None
line_endpoints = None

if lines is not None and len(lines) > 0:
    # Choose the longest detected segment.
    best = max(lines[:, 0, :], key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1]))
    x1, y1, x2, y2 = best
    line_endpoints = (x1, y1, x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length > 0:
        unit_vector = (dx / length, dy / length)
        vector_text = f"Unit vector = ({unit_vector[0]:.3f}, {unit_vector[1]:.3f})"

# If Hough failed, fallback to fitting a line to the largest contour.
if unit_vector is None:
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv.contourArea)
        if cv.contourArea(biggest) > 50:
            [vx, vy, x0, y0] = cv.fitLine(biggest, cv.DIST_L2, 0, 0.01, 0.01)
            if vx != 0 or vy != 0:
                unit_vector = (float(vx), float(vy))
                norm = np.hypot(unit_vector[0], unit_vector[1])
                unit_vector = (unit_vector[0] / norm, unit_vector[1] / norm)
                vector_text = f"Unit vector = ({unit_vector[0]:.3f}, {unit_vector[1]:.3f})"
                # Choose a centered line in the cropped image.
                cx, cy = cropped.shape[1] / 2, cropped.shape[0] / 2
                length = min(cropped.shape[1], cropped.shape[0]) * 0.45
                x1 = int(cx - unit_vector[0] * length)
                y1 = int(cy - unit_vector[1] * length)
                x2 = int(cx + unit_vector[0] * length)
                y2 = int(cy + unit_vector[1] * length)
                line_endpoints = (x1, y1, x2, y2)

if unit_vector is not None and line_endpoints is not None:
    x1, y1, x2, y2 = line_endpoints
    cv.line(line_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=4)
    arrow_tip = (int(x1 + unit_vector[0] * 80), int(y1 + unit_vector[1] * 80))
    cv.arrowedLine(line_img, (x1, y1), arrow_tip, color=(0, 255, 0), thickness=3, tipLength=0.15)
    cv.circle(line_img, (x1, y1), radius=6, color=(0, 255, 0), thickness=-1)
    cv.putText(line_img, vector_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
else:
    cv.putText(line_img, vector_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

# Plot the cropped image and the extracted line/vector overlay.
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(cropped)
ax[0].set_title('Cropped input')
ax[0].axis('off')
ax[1].imshow(line_img)
ax[1].set_title('Detected line and unit vector')
ax[1].axis('off')

if unit_vector is not None:
    print(f"Detected unit vector x,y = {unit_vector[0]:.6f}, {unit_vector[1]:.6f}")

plt.tight_layout()
plt.show()


