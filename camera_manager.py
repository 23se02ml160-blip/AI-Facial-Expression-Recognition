import cv2

class CameraManager:
    def __init__(self):
        self.cap = None
        self.index = 0

    def get_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return self.cap, self.index

        # Try indices 0–4
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cap = cap
                self.index = i
                print(f"[INFO] Camera opened on index {i}")
                return self.cap, i

        print("[ERROR] No camera found")
        return None, None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[INFO] Camera released")

camera_manager = CameraManager()