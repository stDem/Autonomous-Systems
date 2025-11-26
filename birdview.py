import numpy as np
import cv2 as cv
from jetcam.csi_camera import CSICamera


# ============================================================
# Centreline extraction + visualization (Steps 1–7)
# ============================================================

def draw_centreline_from_bev(und, bev, H, src_pts,
                             deg=2,
                             stride=4,
                             min_white_per_row=30,
                             debug=True):
    """
    und      : undistorted original image (BGR)
    bev      : bird's-eye-view image (BGR)
    H        : 3x3 homography (orig -> BEV)
    src_pts  : 4x2 points (orig coords) used for H (ROI polygon)
    deg      : polynomial degree for centreline model
    stride   : row stride when scanning from bottom up
    """

    h_bev, w_bev = bev.shape[:2]

    # ---------- Step 1: BGR -> HSV ----------
    bev_hsv = cv.cvtColor(bev, cv.COLOR_BGR2HSV)

    # ---------- Step 2: Color mask ----------
    # sample a patch near bottom centre to estimate lane colour in HSV
    patch_w = w_bev // 10
    patch_h = h_bev // 10
    x0 = w_bev // 2 - patch_w // 2
    y0 = h_bev - patch_h - 10
    x0 = max(0, x0)
    y0 = max(0, y0)
    patch = bev_hsv[y0:y0 + patch_h, x0:x0 + patch_w]

    h_med = float(np.median(patch[:, :, 0]))
    s_med = float(np.median(patch[:, :, 1]))
    v_med = float(np.median(patch[:, :, 2]))

    dH, dS, dV = 10, 60, 60  # tune if needed
    lower = np.array([
        max(h_med - dH, 0),
        max(s_med - dS, 0),
        max(v_med - dV, 0)
    ], dtype=np.uint8)
    upper = np.array([
        min(h_med + dH, 179),
        min(s_med + dS, 255),
        min(v_med + dV, 255)
    ], dtype=np.uint8)

    color_mask = cv.inRange(bev_hsv, lower, upper)
    color_mask = cv.dilate(color_mask, np.ones((3, 3), np.uint8), iterations=1)

    if debug:
        dbg_bev_color = bev.copy()
        cv.rectangle(dbg_bev_color, (x0, y0), (x0 + patch_w, y0 + patch_h), (0, 255, 255), 2)
        cv.imshow("step2_color_bev_patch", dbg_bev_color)
        cv.imshow("step2_color_mask", color_mask)

    # ---------- Step 3: Gradient mask (Sobel) ----------
    gray = cv.cvtColor(bev, cv.COLOR_BGR2GRAY)
    Gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    Gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(Gx, Gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    tmin, tmax = 30, 255
    grad_mask = cv.inRange(mag, tmin, tmax)

    if debug:
        cv.imshow("step3_grad_mag", mag)
        cv.imshow("step3_grad_mask", grad_mask)

    # ---------- Step 4: Combine masks ----------
    combined = cv.bitwise_and(color_mask, grad_mask)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel, iterations=1)

    if debug:
        cv.imshow("step4_combined_mask", combined)

    # ---------- Step 5: Find centreline points in BEV ----------
    points = []  # (x_mean, y) pairs

    for y in range(h_bev - 1, h_bev // 2, -stride):
        row = combined[y, :]
        xs = np.where(row > 0)[0]
        if xs.size >= min_white_per_row:
            x_mean = float(xs.mean())
            points.append((x_mean, float(y)))

    if debug:
        bev_pts_vis = bev.copy()
        for (x, y) in points:
            cv.circle(bev_pts_vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv.imshow("step5_centre_points_bev", bev_pts_vis)

    if len(points) < deg + 1:
        # not enough data to fit polynomial, just return originals
        return und, bev

    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # ---------- Step 6: Fit polynomial x(y) ----------
    coeffs = np.polyfit(ys, xs, deg)
    poly = np.poly1d(coeffs)

    # evaluate polynomial for all rows in ROI (bottom half)
    centreline_bev = []
    for y in range(h_bev - 1, h_bev // 2, -1):
        x = float(poly(y))
        if 0 <= x < w_bev:
            centreline_bev.append((x, float(y)))

    bev_with_line = bev.copy()
    for (x, y) in centreline_bev:
        cv.circle(bev_with_line, (int(x), int(y)), 1, (255, 0, 0), -1)

    if debug:
        cv.imshow("step6_poly_bev", bev_with_line)

    # ---------- Step 7: Project centreline back to original image ----------
    H_inv = np.linalg.inv(H)

    bev_pts = np.array(centreline_bev, dtype=np.float32)
    ones = np.ones((bev_pts.shape[0], 1), dtype=np.float32)
    bev_hom = np.hstack([bev_pts, ones])  # (N,3)

    orig_hom = bev_hom @ H_inv.T
    xs_o = orig_hom[:, 0] / orig_hom[:, 2]
    ys_o = orig_hom[:, 1] / orig_hom[:, 2]

    und_with_overlay = und.copy()

    # draw ROI polygon (green)
    if src_pts is not None:
        cv.polylines(und_with_overlay,
                     [src_pts.astype(int)],
                     isClosed=True,
                     color=(0, 255, 0),
                     thickness=2)

    # draw centreline (blue) in original image
    for x_o, y_o in zip(xs_o, ys_o):
        x_i = int(round(x_o))
        y_i = int(round(y_o))
        if 0 <= x_i < und.shape[1] and 0 <= y_i < und.shape[0]:
            cv.circle(und_with_overlay, (x_i, y_i), 1, (255, 0, 0), -1)

    return und_with_overlay, bev_with_line


# ============================================================
# Main: Bird's-eye view + live centreline overlay
# ============================================================

def main():
    # ---------- 1) Load calibration ----------
    data = np.load("camera_calib.npz")
    K = data["K"].astype(np.float32)
    dist = data["dist"].astype(np.float32)
    calib_w, calib_h = data["image_size"]

    print("Loaded camera_calib.npz")
    print("Calibration image size:", calib_w, calib_h)
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    # ---------- 2) Start CSI camera ----------
    cam_w, cam_h = 1280, 720

    camera = CSICamera(
        width=cam_w,
        height=cam_h,
        capture_width=cam_w,
        capture_height=cam_h,
        capture_fps=30,
        flip_method=0
    )
    camera.running = True

    # scale K from calibration size -> live size
    sx = cam_w / float(calib_w)
    sy = cam_h / float(calib_h)

    K_live = K.copy()
    K_live[0, 0] *= sx  # fx
    K_live[1, 1] *= sy  # fy
    K_live[0, 2] *= sx  # cx
    K_live[1, 2] *= sy  # cy

    print("Camera size:", cam_w, cam_h)
    print("Scaled K_live:\n", K_live)

    # ---------- 3) Undistortion maps ----------
    newK = K_live
    map1, map2 = cv.initUndistortRectifyMap(
        K_live, dist, None, newK, (cam_w, cam_h), cv.CV_16SC2
    )
    print("Undistort maps ready")

    # ---------- 4) LIVE point selection (TL -> TR -> BR -> BL) ----------
    clicked = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))
            print(f"Point {len(clicked)}: ({x}, {y})")

    select_win = "Select 4 points (TL -> TR -> BR -> BL)"
    cv.namedWindow(select_win, cv.WINDOW_NORMAL)
    cv.resizeWindow(select_win, 960, 540)
    cv.setMouseCallback(select_win, mouse_cb)

    print("=== POINT SELECTION MODE ===")
    print("Click 4 corners on the TRACK in order: TL, TR, BR, BL.")
    print("After you have 4 points, press any key in the window to continue.")
    print("Press ESC to cancel.")

    while True:
        frame_bgr = camera.value
        if frame_bgr is None:
            continue

        und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)

        display = und.copy()
        for i, (x, y) in enumerate(clicked):
            cv.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv.putText(display, str(i + 1), (int(x) + 5, int(y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(select_win, display)
        key = cv.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Cancelled by user (ESC).")
            camera.running = False
            cv.destroyAllWindows()
            return

        if len(clicked) == 4 and key != 255:
            break

    cv.destroyWindow(select_win)

    if len(clicked) != 4:
        print("ERROR: need exactly 4 points, got", len(clicked))
        camera.running = False
        cv.destroyAllWindows()
        return

    src_pts = np.array(clicked, dtype=np.float32)
    print("Selected src_pts:\n", src_pts)

    # ---------- 5) Build homography for bird’s-eye ----------
    bev_w, bev_h = 500, 800
    dst_pts = np.array([
        [0,      0],      # TL
        [bev_w,  0],      # TR
        [bev_w,  bev_h],  # BR
        [0,      bev_h],  # BL
    ], dtype=np.float32)

    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    print("Homography H:\n", H)

    # ---------- 6) Live bird’s-eye + centreline ----------
    print("=== LIVE BIRD-VIEW + CENTRELINE MODE ===")
    print("Press ESC to exit.")

    try:
        while True:
            frame_bgr = camera.value
            if frame_bgr is None:
                continue

            und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)
            bev = cv.warpPerspective(und, H, (bev_w, bev_h), flags=cv.INTER_LINEAR)

            # Extract & draw centreline in BEV and original
            und_vis, bev_vis = draw_centreline_from_bev(
                und, bev, H, src_pts,
                deg=2,
                stride=4,
                min_white_per_row=30,
                debug=False   # set True to see intermediate steps
            )

            cv.imshow("Undistorted + Centreline", und_vis)
            cv.imshow("Bird View + Centreline", bev_vis)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        camera.running = False
        cv.destroyAllWindows()
        print("Stopped camera and closed windows")


if __name__ == "__main__":
    main()
