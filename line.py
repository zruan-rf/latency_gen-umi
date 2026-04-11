import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = "/home/roboforce/Desktop/gen-umi/20260410_173539/cam4_24276935100461.jpg"
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cropped = img[200:800, 300:1300]

# Convert to grayscale and detect edges for line extraction.
gray = cv.cvtColor(cropped, cv.COLOR_RGB2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Try probabilistic Hough transform first.
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=60, minLineLength=80, maxLineGap=20)

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
    cv.putText(line_img, vector_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
else:
    cv.putText(line_img, vector_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

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


