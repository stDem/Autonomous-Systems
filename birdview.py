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
    # print("Calibration image size:", calib_w, calib_h)
    # print("RMS reprojection error:", rms)
    # print("K:\n", K)
    # print("dist:\n", dist.ravel())

    # ---------- 2) Start CSI camera at 224x224 ----------
    cam_w, cam_h = 224, 224
    camera = CSICamera(width=cam_w, height=cam_h)
    camera.running = True
    print("CSI camera started at", cam_w, "x", cam_h)

    # ---------- 3) Scale intrinsics from calib size -> live size ----------
    sx = cam_w / float(calib_w)
    sy = cam_h / float(calib_h)
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy

    # print("Scaled K for live size:\n", K_scaled)

    newK, _ = cv.getOptimalNewCameraMatrix(K_scaled, dist, (cam_w, cam_h), alpha=0)
    map1, map2 = cv.initUndistortRectifyMap(
        K_scaled, dist, None, newK, (cam_w, cam_h), cv.CV_16SC2
    )
    # print("Undistort maps ready")

    # ---------- 4) Capture one UNDISTORTED frame for point selection ----------
    # print("Capturing frame for point selection...")
    frame_rgb = None
    while frame_rgb is None:
        frame_rgb = camera.value

    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    if frame_bgr.shape[1] != cam_w or frame_bgr.shape[0] != cam_h:
        frame_bgr = cv.resize(frame_bgr, (cam_w, cam_h), interpolation=cv.INTER_AREA)

    und_preview = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)

    # we will draw on a copy so the original und_preview stays valid
    draw_img = und_preview.copy()

    # ---------- 5) Let user click 4 points (TL -> TR -> BR -> BL) ----------
    clicked = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal draw_img
        if event == cv.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))
            print(f"Point {len(clicked)}: ({x}, {y})")
            # draw a small circle and label on the image
            cv.circle(draw_img, (x, y), 4, (0, 255, 0), -1)
            cv.putText(draw_img, str(len(clicked)), (x + 5, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.imshow("Select 4 points (TL -> TR -> BR -> BL)", draw_img)

    win_name = "Select 4 points (TL -> TR -> BR -> BL)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, mouse_cb)

    # print("Click 4 corners on the track (TL, TR, BR, BL).")
    # print("When done, press any key in the window to continue.")

    while True:
        cv.imshow(win_name, draw_img)
        key = cv.waitKey(20) & 0xFF
        # After 4 points, any key continues
        if key != 255 and len(clicked) == 4:
            break
        # ESC cancels
        if key == 27:
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
    # print("Selected src_pts:\n", src_pts)

    # ---------- 6) Build homography for bird’s-eye ----------
    bev_w, bev_h = 300, 400  # bird view output size
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
            if frame_bgr.shape[1] != cam_w or frame_bgr.shape[0] != cam_h:
                frame_bgr = cv.resize(frame_bgr, (cam_w, cam_h), interpolation=cv.INTER_AREA)

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
        # print("Stopped camera and closed windows")


if __name__ == "__main__":
    main()
