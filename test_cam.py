from jetcam.csi_camera import CSICamera
import cv2 as cv

camera = CSICamera(width=1280, height=720, capture_width=1280, capture_height=720,
                   capture_fps=30, flip_method=0)
camera.running = True

try:
    while True:
        frame = camera.value
        if frame is None:
            continue
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("CSI Camera", frame_bgr)
        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break
finally:
    camera.running = False
    cv.destroyAllWindows()

