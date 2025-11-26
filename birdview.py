import numpy as np
import cv2 as cv
from jetcam.csi_camera import CSICamera


def main():
    # ---------- 1) Load calibration ----------
    data = np.load("camera_calib.npz")
    K = data["K"].astype(np.float32)
    dist = data["dist"].astype(np.float32)
    calib_w, calib_h = data["image_size"]  # [width, height]
    rms = data["rms"]

    print("Loaded camera_calib.npz")
    print("Calibration image size:", calib_w, calib_h)
    print("RMS reprojection error:", rms)
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    # ---------- 2) Start CSI camera at CALIBRATION SIZE ----------
    # Important: use same aspect ratio/resolution as calibration images.
    cam_w, cam_h = int(calib_w), int(calib_h)   # e.g. 1280x720

    camera = CSICamera(
        width=cam_w,
        height=cam_h,
        capture_width=cam_w,
        capture_height=cam_h,
        capture_fps=30,
        flip_method=0
    )
    camera.running = True
    print("CSI camera started at", cam_w, "x", cam_h)

    # ---------- 3) Intrinsics for live size ----------
    # We calibrated at (calib_w, calib_h), and we run camera at same size,
    # so we can just use K directly (no scaling needed).
    K_live = K.copy()

    newK, _ = cv.getOptimalNewCameraMatrix(K_live, dist, (cam_w, cam_h), alpha=0)
    map1, map2 = cv.initUndistortRectifyMap(
        K_live, dist, None, newK, (cam_w, cam_h), cv.CV_16SC2
    )
    print("Undistort maps ready")

    # ---------- 4) Capture one UNDISTORTED frame for point selection ----------
    print("Capturing frame for point selection...")
    frame_rgb = None
    while frame_rgb is None:
        frame_rgb = camera.value

    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    und_preview = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)

    # We will draw on a copy so the original stays valid
    draw_img = und_preview.copy()

    # ---------- 5) Let user click 4 points (TL -> TR -> BR -> BL) ----------
    clicked = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal draw_img
        if event == cv.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))
            print(f"Point {len(clicked)}: ({x}, {y})")
            cv.circle(draw_img, (x, y), 6, (0, 255, 0), -1)
            cv.putText(draw_img, str(len(clicked)), (x + 5, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow(win_name, draw_img)

    win_name = "Select 4 points (TL -> TR -> BR -> BL)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, mouse_cb)

    print("Click 4 corners on the track in order: TL, TR, BR, BL.")
    print("When done, press any key in the image window to continue.")

    while True:
        cv.imshow(win_name, draw_img)
        key = cv.waitKey(20) & 0xFF
        # After 4 points, any key continues
        if key != 255 and len(clicked) == 4:
            break
        if key == 27:  # ESC cancels
            print("Cancelled by user (ESC).")
            camera.running = False
            cv.destroyAllWindows()
            return

    cv.destroyWindow(win_name)

    if len(clicked) != 4:
        print("ERROR: need exactly 4 points, got", len(clicked))
        camera.running = False
        cv.destroyAllWindows()
        return

    src_pts = np.array(clicked, dtype=np.float32)
    print("Selected src_pts:\n", src_pts)

    # ---------- 6) Build homography for bird’s-eye ----------
    bev_w, bev_h = 500, 800  # bird view output size; tweak as you like
    dst_pts = np.array([
        [0,      0],      # TL
        [bev_w,  0],      # TR
        [bev_w,  bev_h],  # BR
        [0,      bev_h],  # BL
    ], dtype=np.float32)

    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    print("Homography H:\n", H)

    # ---------- 7) Live loop: undistorted + bird’s-eye ----------
    try:
        while True:
            frame_rgb = camera.value
            if frame_rgb is None:
                continue

            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

            und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)
            bev = cv.warpPerspective(und, H, (bev_w, bev_h), flags=cv.INTER_LINEAR)

            # If the windows appear too small on your Mac,
            # you can upscale just for display (doesn't change the math):
            display_und = und  # or cv.resize(und, None, fx=0.7, fy=0.7)
            display_bev = bev  # or cv.resize(bev, None, fx=1.0, fy=1.0)

            cv.imshow("Undistorted", display_und)
            cv.imshow("Bird View", display_bev)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        camera.running = False
        cv.destroyAllWindows()
        print("Stopped camera and closed windows")


if __name__ == "__main__":
    main()
