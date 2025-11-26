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

    # ---------- 2) Start CSI camera at calibration size ----------
    cam_w, cam_h = int(calib_w), int(calib_h)   # use same size as chessboard images

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

    # ---------- 3) Undistortion maps (no scaling, same size as calibration) ----------
    K_live = K.copy()
    newK, _ = cv.getOptimalNewCameraMatrix(K_live, dist, (cam_w, cam_h), alpha=0)
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
        frame_rgb = camera.value
        if frame_rgb is None:
            continue

        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)

        # draw already-clicked points
        display = und.copy()
        for i, (x, y) in enumerate(clicked):
            cv.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv.putText(display, str(i + 1), (int(x) + 5, int(y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(select_win, display)
        key = cv.waitKey(20) & 0xFF

        if key == 27:  # ESC cancels
            print("Cancelled by user (ESC).")
            camera.running = False
            cv.destroyAllWindows()
            return

        # once 4 points are clicked, any key continues
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
    bev_w, bev_h = 500, 800  # adjust if you want different aspect/size
    dst_pts = np.array([
        [0,      0],      # TL
        [bev_w,  0],      # TR
        [bev_w,  bev_h],  # BR
        [0,      bev_h],  # BL
    ], dtype=np.float32)

    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    print("Homography H:\n", H)

    # ---------- 6) Live bird’s-eye view ----------
    cv.namedWindow("Undistorted", cv.WINDOW_NORMAL)
    cv.resizeWindow("Undistorted", 960, 540)
    cv.namedWindow("Bird View", cv.WINDOW_NORMAL)
    cv.resizeWindow("Bird View", 600, 800)

    print("=== LIVE BIRD-VIEW MODE ===")
    print("Press ESC to exit.")

    try:
        while True:
            frame_rgb = camera.value
            if frame_rgb is None:
                continue

            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)
            bev = cv.warpPerspective(und, H, (bev_w, bev_h), flags=cv.INTER_LINEAR)

            cv.imshow("Undistorted", und)
            cv.imshow("Bird View", bev)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        camera.running = False
        cv.destroyAllWindows()
        print("Stopped camera and closed windows")


if __name__ == "__main__":
    main()
