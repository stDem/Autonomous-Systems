import numpy as np
import cv2 as cv
import glob
import sys

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points for an 8x5 chessboard
objp = np.zeros((5 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

objpoints = []   # 3D points in real world space
imgpoints = []   # 2D points in image plane

# Load chessboard images
images = glob.glob('./chess/*.jpg')
# print("Found images:", len(images))

if len(images) == 0:
    print("No images found in ./chess/")
    sys.exit(1)

# Process each image
for fname in images:
    img = cv.imread(fname)
    if img is None:
        print("Could not read:", fname)
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8, 5), None)

    # print(fname, "ret =", ret)

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Ensure we found at least some corners
if len(objpoints) == 0:
    # print("No chessboard corners detected. Cannot calibrate.")
    sys.exit(1)

# ====== CAMERA CALIBRATION ======
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("RMS reprojection error:", ret)
print("Camera matrix (K):\n", mtx)
print("Distortion coefficients (dist):\n", dist.ravel())

# Save calibration results
image_size = gray.shape[::-1]  # (width, height)

np.savez(
    "camera_calib.npz",
    K=mtx,
    dist=dist,
    image_size=np.array(image_size),
    rms=ret
)

print("Saved camera_calib.npz")
