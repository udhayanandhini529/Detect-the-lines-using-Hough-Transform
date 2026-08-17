#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```python
# Read the image using OpenCV
img = cv2.imread('lan_img1.jpg')
```

---

###  Step 3: Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

###  Step 4: Display Images

```python
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.show()
```

---

###  Step 5: Thresholding

```python
# Apply thresholding

threshold = 
cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)[1]


# ============================================================
# Display Threshold Image
# ============================================================

plt.figure(figsize=(20, 10))
plt.subplot(1, 1, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Threshold')
plt.axis('off')
plt.show()

```

---

###  Step 6: Region of Interest (ROI)

```python
# Region masking: Select vertices according to the input image.
roi_vertices = np.array([[[100, 540],
                          [900, 540],
                          [515, 320],
                          [450, 320]]])

# Defining a blank mask.
mask = np.zeros_like(threshold)

# Defining a 3 channel or 1 channel color to fill the mask.
if len(threshold.shape) > 2:
    channel_count = threshold.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

# Filling pixels inside the polygon.
cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

# Constructing the region of interest.
roi = cv2.bitwise_and(threshold, mask)

# Display ROI images.
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Initial threshold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Polyfill mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(roi, cmap='gray')
plt.title('Isolated roi')
plt.axis('off')

plt.show()
```

---

### Step 7: Edge Detection (Canny)

```python
edges = cv2.Canny(roi, 50, 150)
```

---

###  Step 8: Gaussian Blur

```python
canny_blur = cv2.GaussianBlur(edges, (5, 5), 0)


# ============================================================
# Display Edge Detection and Blurred Images
# ============================================================

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(canny_blur, cmap='gray')
plt.title('Blurred edges')
plt.axis('off')

plt.show()

```

---

###  Step 9: Hough Transform

```python
# Hough transform parameters
lines = cv2.HoughLinesP(
    canny_blur,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=100
)


# ============================================================
# Utility Function for Drawing Lines
# ============================================================

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# ============================================================
# Display Hough Lines
# ============================================================

hough = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

draw_lines(hough, lines)

if lines is not None:
    print(
        "Found {} lines, including: {}".format(
            len(lines), lines[0]
        )
    )
else:
    print("No lines found.")

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB))
plt.title('Hough Lines')
plt.axis('off')
plt.show()
```

---

### Step 10: Lane Detection Logic

```python
def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:

                if y1 > y2:
                    # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])

                elif y1 < y2:
                    # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])

    return left_lines, right_lines


# ============================================================
# Calculate Average
# ============================================================

def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1

        return sum(values) / n


# ============================================================
# Extrapolate Lane Lines
# ============================================================

def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""

    slopes = []
    consts = []

    if lines is not None:
        for x1, y1, x2, y2 in lines:

            slope = (y1 - y2) / (x1 - x2)
            slopes.append(slope)

            c = y1 - slope * x1
            consts.append(c)

    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)

    # Calculate average intersection at lower_border.
    x_lane_lower_point = int(
        (lower_border - avg_consts) / avg_slope
    )

    # Calculate average intersection at upper_border.
    x_lane_upper_point = int(
        (upper_border - avg_consts) / avg_slope
    )

    return [
        x_lane_lower_point,
        lower_border,
        x_lane_upper_point,
        upper_border
    ]


# ============================================================
# Lane Detection Logic
# ============================================================

# Define bounds of the region of interest.
roi_upper_border = 340
roi_lower_border = 540

# Create a blank array to contain the colorized results.
lanes_img = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

# Identify lists of left-sided and right-sided lines.
lines_left, lines_right = separate_left_right_lines(lines)

# Extrapolate the lists of lines into recognized lanes.
lane_left = extrapolate_lines(
    lines_left,
    roi_upper_border,
    roi_lower_border
)

lane_right = extrapolate_lines(
    lines_right,
    roi_upper_border,
    roi_lower_border
)

# Draw the detected lanes.
draw_lines(
    lanes_img,
    [[lane_left]],
    thickness=10
)

draw_lines(
    lanes_img,
    [[lane_right]],
    thickness=10
)


# ============================================================
# Display Final Results
# ============================================================

fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB))
ax.set_title('Before extrapolation')
ax.axis('off')

ax = fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(lanes_img, cv2.COLOR_BGR2RGB))
ax.set_title('After extrapolation')
ax.axis('off')

plt.show()
```

---

##  Expected Output

* Original image
![alt text](image.png)
* Grayscale image
![alt text](image-2.png)
* Thresholded image
![alt text](image-3.png)
![alt text](image-4.png)
* ROI masked image
![alt text](image-5.png)
* Edge detected image
![alt text](image-6.png)
* Smoothed image
![alt text](image-7.png)
* Detected lines
![alt text](image-8.png)
* Final lane detection output
![alt text](image-9.png)
---

##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name:** ____________________________
* **Register No:** ______________________
