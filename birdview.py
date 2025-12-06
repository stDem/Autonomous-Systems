import numpy as np
import cv2 as cv
from jetcam.csi_camera import CSICamera
import time 
from jetracer.nvidia_racecar import NvidiaRacecar

# ============================================================
# Centreline extraction + visualization (Steps 1–7)
# ============================================================

def draw_centreline_from_bev(
    und,
    bev,
    H,
    src_pts,
    deg=2,
    stride=4,
    min_white_per_row=10,
    debug=False,
    save_debug_prefix=None,
):
    """
    und      : undistorted original image (BGR)
    bev      : bird's-eye-view image (BGR)
    H        : 3x3 homography (orig -> BEV)
    src_pts  : 4x2 points (orig coords) used for H (ROI polygon)
    deg      : polynomial degree for centreline model
    stride   : row stride when scanning from bottom up
    debug    : if True, show intermediate images with cv.imshow
    save_debug_prefix: if not None, save PNGs for each step, e.g. 'debug_frame'

    Returns:
        und_with_overlay : original view with ROI + centreline
        bev_with_line    : BEV view with centreline
        coeffs           : np.array of polynomial coefficients (or None)
        points           : list of (x_mean, y) points used to fit poly
    """

    h_bev, w_bev = bev.shape[:2]

    # ---------------- Step 1: BGR -> HSV ----------------
    bev_hsv = cv.cvtColor(bev, cv.COLOR_BGR2HSV)
    h_bev, w_bev = bev_hsv.shape[:2]

    # ---------------- Step 2: Color mask based on centreline HSV ----------------
    # 2a) Choose a patch where we EXPECT the centreline (bottom–centre of BEV)
    #    tune these fractions if your line is shifted
    patch_w_frac = 0.15   # width of patch as fraction of image width
    patch_h_frac = 0.25   # height of patch as fraction of image height

    x0 = int(w_bev * (0.5 - patch_w_frac / 2.0))
    x1 = int(w_bev * (0.5 + patch_w_frac / 2.0))
    y0 = int(h_bev * (1.0 - patch_h_frac))
    y1 = h_bev

    patch = bev_hsv[y0:y1, x0:x1]

    # 2b) Keep only bright pixels in that patch (they are likely the line)
    patch_flat = patch.reshape(-1, 3)
    V_patch = patch_flat[:, 2]
    bright_mask = V_patch > (V_patch.mean() + 5)  # "+5" is a small offset; tune if needed

    if np.any(bright_mask):
        centre_pixels = patch_flat[bright_mask]
    else:
        centre_pixels = patch_flat  # fallback: use whole patch

    mean_hsv = centre_pixels.mean(axis=0)
    std_hsv  = centre_pixels.std(axis=0)

    Hm, Sm, Vm = mean_hsv
    s_std, v_std = std_hsv[1], std_hsv[2]

    # 2c) Build HSV bounds around the centreline colour
    # If S is very small, the line is nearly grey -> ignore H (use full [0,179])
    if Sm < 15:
        h_low, h_high = 0, 179
    else:
        dH = 15
        h_low  = max(0,   Hm - dH)
        h_high = min(179, Hm + dH)

    # Saturation: narrow range around Sm
    dS = max(10, 2 * s_std)          # at least +-10, or wider if noisy
    s_low  = max(0,   Sm - dS)
    s_high = min(255, Sm + dS)

    # Value (brightness): we know the line is bright
    dV = max(15, 2 * v_std)
    v_low  = max(0,   Vm - dV)
    v_high = 255      # allow up to max

    lower_white = np.array([h_low,  s_low,  v_low ], dtype=np.uint8)
    upper_white = np.array([h_high, s_high, v_high], dtype=np.uint8)

    # DEBUG: you can print these once to see what they are
    # print("centreline HSV mean:", mean_hsv, "lower:", lower_white, "upper:", upper_white)

    # 2d) Apply inRange with these dynamic bounds
    color_mask = cv.inRange(bev_hsv, lower_white, upper_white)

    kernel_small = np.ones((3, 3), np.uint8)
    # remove tiny speckles
    color_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel_small, iterations=1)
    # keep the line continuous
    color_mask = cv.dilate(color_mask, kernel_small, iterations=1)

    if debug:
        cv.imshow("step2_color_mask", color_mask)
    if save_debug_prefix is not None:
        cv.imwrite(f"{save_debug_prefix}_step2_color_mask.png", color_mask)


    # ---------------- Step 3: Gradient mask (Sobel) ----------------
    gray = cv.cvtColor(bev, cv.COLOR_BGR2GRAY)
    Gx = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3)
    absGx = cv.convertScaleAbs(Gx)

    # threshold for edges
    tmin = 20            # was 30
    grad_mask = cv.inRange(absGx, tmin, 255)

    if debug:
        # cv.imshow("step3_grad_absGx", absGx)
        cv.imshow("step3_grad_mask", grad_mask)
    if save_debug_prefix is not None:
        # cv.imwrite(f"{save_debug_prefix}_step3_grad_absGx.png", absGx)
        cv.imwrite(f"{save_debug_prefix}_step3_grad_mask.png", grad_mask)

    # ---------------- Step 4: Combine masks ----------------
    USE_GRADIENT = True

    if USE_GRADIENT:
        combined = cv.bitwise_and(color_mask, grad_mask)
    else:
        combined = color_mask.copy()

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel, iterations=1)

    if debug:
        cv.imshow("step4_combined_mask", combined)
    if save_debug_prefix is not None:
        cv.imwrite(f"{save_debug_prefix}_step4_combined_mask.png", combined)

    # ---------------- Step 5: Find centreline points (row-wise) ----------------
    points = []  # (c, r) = (x, y)

    # optional: central ROI to ignore walls; keep wide for now
    roi_x_min = int(w_bev * 0.05)
    roi_x_max = int(w_bev * 0.95)

    # how many white pixels needed in a row to trust it
    min_white_per_row = 3   # much smaller than before – easier to collect points

    for y in range(h_bev - 1, h_bev // 2, -stride):
        row = combined[y, :]

        # restrict to central ROI
        row_roi = row[roi_x_min:roi_x_max]
        xs_roi = np.where(row_roi > 0)[0]  # indices inside ROI

        if xs_roi.size < min_white_per_row:
            # not enough white pixels -> skip this row
            continue

        # convert ROI indices to full-image x positions
        xs = xs_roi + roi_x_min

        # assignment spec: c = average column index of white pixels
        c = float(xs.mean())

        points.append((c, float(y)))

    # visualisation of chosen points
    bev_pts_vis = bev.copy()
    for (x, y) in points:
        cv.circle(bev_pts_vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    if debug:
        cv.imshow("step5_centre_points_bev", bev_pts_vis)
    if save_debug_prefix is not None:
        cv.imwrite(f"{save_debug_prefix}_step5_centre_points_bev.png", bev_pts_vis)

    if len(points) < deg + 1:
        # not enough data – return original images unchanged, no polynomial
        return und, bev, None, points

    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # ---------------- Step 6: Polynomial fit x(y) ----------------
    coeffs = np.polyfit(ys, xs, deg)
    poly = np.poly1d(coeffs)

    centreline_bev = []
    for y in range(h_bev - 1, h_bev // 2, -1):
        x = float(poly(y))
        if 0 <= x < w_bev:
            centreline_bev.append((x, float(y)))

    bev_with_line = bev.copy()
    for (x, y) in centreline_bev:
        cv.circle(bev_with_line, (int(x), int(y)), 1, (255, 0, 0), -1)

    # if debug:
    #     cv.imshow("step6_poly_bev", bev_with_line)
    # if save_debug_prefix is not None:
    #     cv.imwrite(f"{save_debug_prefix}_step6_poly_bev.png", bev_with_line)

    # ---------------- Step 7: Project centreline back to original ----------------
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
        cv.polylines(
            und_with_overlay,
            [src_pts.astype(int)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    # draw centreline (blue) in original image
    for x_o, y_o in zip(xs_o, ys_o):
        x_i = int(round(x_o))
        y_i = int(round(y_o))
        if 0 <= x_i < und.shape[1] and 0 <= y_i < und.shape[0]:
            cv.circle(und_with_overlay, (x_i, y_i), 1, (255, 0, 0), -1)

    if save_debug_prefix is not None:
        cv.imwrite(f"{save_debug_prefix}_step7_und_with_centreline.png", und_with_overlay)

    return und_with_overlay, bev_with_line, coeffs, points


def compute_max_speed_from_poly(
    coeffs,
    h_bev,
    y_min_fraction=2/3,   # lower third of the image
    y_max_fraction=1.0,
    meters_per_pixel_y=0.01,
    meters_per_pixel_x=0.01,
    mu=0.5,
    g=9.81,
    kappa_clip=5.0,
):
    """
    Compute maximum allowed speed based on lane curvature.

    coeffs : polynomial coefficients for x(y) in pixel coordinates
    h_bev  : BEV image height in pixels
    y_min_fraction, y_max_fraction : which y-region to inspect (e.g. lower third)
    meters_per_pixel_* : pixel -> meter scaling (approximate)
    mu   : friction/grip coefficient (linoleum ~ 0.5)
    g    : gravity [m/s^2]
    kappa_clip : clamp curvature to avoid crazy outliers

    Returns:
        v_max_mps : max speed [m/s]
        kappa_max : max curvature [1/m]
        R_min     : min radius [m]
    """

    if coeffs is None:
        return 0.0, 0.0, float("inf")

    poly = np.poly1d(coeffs)
    dpoly = np.polyder(poly, 1)
    ddpoly = np.polyder(poly, 2)

    y0 = int(h_bev * y_min_fraction)
    y1 = int(h_bev * y_max_fraction)

    curvatures = []
    for y_pix in range(y0, y1):
        x_prime_pix = dpoly(y_pix)
        x_dprime_pix = ddpoly(y_pix)

        # convert derivatives from pixels to meters
        dy_m = meters_per_pixel_y
        dx_m = meters_per_pixel_x

        # dx/dy in meters per meter (unitless slope)
        x_prime = (dx_m / dy_m) * x_prime_pix
        x_dprime = (dx_m / (dy_m ** 2)) * x_dprime_pix

        # curvature κ = |x''| / (1 + x'^2)^(3/2)
        kappa = abs(x_dprime) / (1.0 + x_prime ** 2) ** 1.5  # [1/m]
        curvatures.append(kappa)

    if not curvatures:
        return 0.0, 0.0, float("inf")

    kappa_max = max(curvatures)
    kappa_max = min(kappa_max, kappa_clip)  # clip outliers

    if kappa_max < 1e-6:  # essentially straight
        R_min = float("inf")
        v_max = 3.0  # some default maximum speed [m/s]; tune as desired
    else:
        R_min = 1.0 / kappa_max
        # lateral acceleration constraint: a_lat = v^2 / R <= mu * g
        # => v_max = sqrt(mu * g * R)
        v_max = (mu * g * R_min) ** 0.5

    return float(v_max), float(kappa_max), float(R_min)





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ==============================================================================================
# Main: Bird's-eye view + live centreline overlay
# ==============================================================================================

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
    
    # ---------- 5b) Create JetRacer car (AUTONOMOUS CONTROL) ----------
    car = NvidiaRacecar()
    car.throttle = 0.0
    car.steering = 0.0

    # set gains like you did in the gamepad example
    car.throttle_gain = 0.25      # how aggressive throttle is (start small)
    car.steering_offset = -0.18   # your known offset
    # car.steering_gain  # you can also tune this if needed

    input("Place the car on the track and press ENTER to start autonomous mode...")


    # ---------- 6) Live bird’s-eye + centreline + speed + steering ----------
    print("=== LIVE BIRD-VIEW + CENTRELINE + SPEED + STEERING MODE ===")
    print("Press ESC to exit.")

    frame_idx = 0

    # EMA (Exponential Moving Average) of polynomial coefficients
    coeffs_ema = None
    ema_alpha = 0.8  # closer to 1.0 = slower changes

    # EMA of speed (for smooth commands)
    speed_ema = 0.0
    speed_alpha = 0.8

    # ---- PID steering state & parameters (all <<< NEW) ----
    Kp = 0.6    # start point; you will tune
    Ki = 0.0
    Kd = 0.1

    T_s = 1.0 / 30.0       # assume 30 FPS; fine for now
    e_prev = 0.0
    I_term = 0.0
    I_max = 2000.0         # integral clamp

    steer_max = 1.0        # normalized command in [-1, 1]
    steer_cmd = 0.0
    steer_ema = 0.0
    steer_alpha = 0.7      # smoothing for steering

    # histories for plotting
    speed_history = []
    steer_history = []      # <<< NEW
    graph_h, graph_w = 200, 400  # pixels of graph window

    try:
        while True:
            frame_bgr = camera.value
            if frame_bgr is None:
                continue

            und = cv.remap(frame_bgr, map1, map2, interpolation=cv.INTER_LINEAR)
            bev = cv.warpPerspective(und, H, (bev_w, bev_h), flags=cv.INTER_LINEAR)

            # For the first frame, enable debug + save PNGs
            if frame_idx == 0:
                und_vis, bev_vis, coeffs, points = draw_centreline_from_bev(
                    und, bev, H, src_pts,
                    deg=2,
                    stride=4,
                    min_white_per_row=20,
                    debug=True,
                    save_debug_prefix="debug_bev"
                )
            else:
                und_vis, bev_vis, coeffs, points = draw_centreline_from_bev(
                    und, bev, H, src_pts,
                    deg=2,
                    stride=4,
                    min_white_per_row=20,
                    debug=False,
                    save_debug_prefix=None
                )

            frame_idx += 1

            # --------- Lane confidence (how many rows had valid points) ---------
            max_rows = (bev_h - bev_h // 2) // 4  # approx for stride=4
            if max_rows > 0:
                confidence = min(1.0, len(points) / max_rows)
            else:
                confidence = 0.0

            # --------- Smooth polynomial coefficients (EMA) ---------
            if coeffs is not None:
                if coeffs_ema is None:
                    coeffs_ema = coeffs
                else:
                    coeffs_ema = ema_alpha * coeffs_ema + (1.0 - ema_alpha) * coeffs

            # --------- Compute curvature-based max speed ---------
            if coeffs_ema is not None:
                v_max, kappa_max, R_min = compute_max_speed_from_poly(
                    coeffs_ema,
                    h_bev=bev_h,
                    y_min_fraction=2/3,   # lower third of BEV
                    y_max_fraction=1.0,
                    meters_per_pixel_y=0.01,
                    meters_per_pixel_x=0.01,
                    mu=0.5,
                    g=9.81,
                    kappa_clip=5.0,
                )
            else:
                v_max, kappa_max, R_min = 0.0, 0.0, float("inf")

            # --------- Safety: scale speed by lane confidence ---------
        # ===============================================
            v_max_clip = 2.0      # how we clip v_conf, you already found this
            v_max_physical = 3.0  # used for scaling speed -> throttle

            MIN_THROTTLE = 0.12   # experimentally: smallest value where wheels start moving
        # ==============================================
            v_conf = v_max * confidence

            # Clip speed to a reasonable range for your car
            v_conf = max(0.0, min(v_conf, v_max_clip))   # [m/s], tune 3.0 as your top speed

            # --------- Smooth speed (momentum term) ---------
            if speed_history:
                speed_ema = speed_alpha * speed_ema + (1.0 - speed_alpha) * v_conf
            else:
                speed_ema = v_conf

            # --------- STEERING: PID based on centreline error (<<< NEW) ---------
            if coeffs_ema is not None:
                poly_ema = np.poly1d(coeffs_ema)

                # choose row near bottom (close to car)
                y_ref = int(bev_h * 0.9)
                x_cl = float(poly_ema(y_ref))     # PV[k] in pixels

                SP = bev_w / 2.0                  # setpoint = image centre
                e_px = SP - x_cl                  # error in pixels (SP - PV)

                # normalize error to [-1, 1] approx (optional but easier to tune)
                e = e_px / (bev_w / 2.0)
            else:
                e_px = 0.0
                e = 0.0

            # PID terms
            P_term = Kp * e
            I_term += Ki * e * T_s
            I_term = max(-I_max, min(I_term, I_max))  # clamp integral
            D_term = Kd * (e - e_prev) / T_s
            e_prev = e

            u = P_term + I_term + D_term

            # saturate steering
            if u > steer_max:
                steer_cmd = steer_max
            elif u < -steer_max:
                steer_cmd = -steer_max
            else:
                steer_cmd = u

            # smooth steering command with EMA
            if steer_history:
                steer_ema = steer_alpha * steer_ema + (1.0 - steer_alpha) * steer_cmd
            else:
                steer_ema = steer_cmd

            # --------- Convert speed_ema (m/s) → throttle [0,1] and send commands ---------
           
            # --------- Convert speed_ema (m/s) → throttle [0,1] and send commands ---------
            if confidence < 0.2 or coeffs_ema is None:
                # lane not reliable → stop and centre steering
                throttle_cmd = 0.0
                steer_to_send = 0.0
            else:
                # simple linear mapping speed -> throttle
                throttle_cmd = speed_ema / v_max_physical

                # ensure we overcome motor dead-zone if we *want* to move
                if throttle_cmd > 0.0:
                    throttle_cmd = max(throttle_cmd, MIN_THROTTLE)

                throttle_cmd = min(throttle_cmd, 1.0)

                # steering is already normalized [-1,1] from PID
                steer_to_send = float(steer_ema)
                steer_to_send = max(-1.0, min(steer_to_send, 1.0))

            car.throttle = float(throttle_cmd)
            car.steering = float(steer_to_send)



            # --------- Store histories for plotting ---------
            speed_history.append(speed_ema)
            steer_history.append(steer_ema)
            if len(speed_history) > graph_w:
                speed_history.pop(0)
            if len(steer_history) > graph_w:
                steer_history.pop(0)

            # --------- Draw combined speed + steering graph (<<< NEW) ---------
            graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)

            # axes
            cv.line(graph, (0, graph_h//2), (graph_w-1, graph_h//2), (255, 255, 255), 1)
            cv.line(graph, (0, graph_h-1), (graph_w-1, graph_h-1), (255, 255, 255), 1)
            cv.line(graph, (0, 0), (0, graph_h-1), (255, 255, 255), 1)

            # top half: speed
            max_speed_for_graph = 3.0
            for i, v in enumerate(speed_history[-graph_w:]):
                x = i
                y = int((graph_h//2 - 5) - (v / max_speed_for_graph) * (graph_h//2 - 10))
                y = np.clip(y, 0, graph_h//2 - 1)
                graph[y:, x] = (0, 255, 0)

            # bottom half: steering [-1,1]
            for i, s in enumerate(steer_history[-graph_w:]):
                x = i
                y_mid = (3 * graph_h) // 4
                y_range = (graph_h // 2 - 10)
                y = int(y_mid - s * y_range)
                y = np.clip(y, graph_h//2, graph_h-1)
                graph[y:, x] = (255, 0, 0)

            cv.putText(graph, "speed (m/s)", (5, 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(graph, "steer (-1..1)", (5, graph_h//2 + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # --------- Text overlay on undistorted image ---------
            cv.putText(und_vis, f"v_cmd = {speed_ema:.2f} m/s", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv.putText(und_vis, f"confidence = {confidence:.2f}", (20, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(und_vis, f"kappa_max = {kappa_max:.2f} 1/m", (20, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(und_vis, f"steer = {steer_ema:.2f}", (20, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(und_vis, f"e_px = {e_px:.1f}", (20, 160),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --------- Show windows ---------
            cv.imshow("Undistorted + Centreline + Speed + Steering", und_vis)
            cv.imshow("Bird View + Centreline", bev_vis)
            cv.imshow("Speed & Steering history", graph)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        camera.running = False
        car.throttle = 0.0    
        car.steering = 0.0 
        cv.destroyAllWindows()
        print("Stopped camera and closed windows")


if __name__ == "__main__":
    main()
