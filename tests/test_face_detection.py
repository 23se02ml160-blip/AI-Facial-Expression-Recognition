import unittest
import cv2
import numpy as np

from utils import detect_faces


class FaceDetectionTests(unittest.TestCase):
    def test_detect_faces_finds_simple_face_like_shape(self):
        img = np.zeros((220, 220, 3), dtype=np.uint8)
        img[:] = 255
        cv2.ellipse(img, (110, 115), (50, 70), 0, 0, 360, (220, 200, 180), -1)
        cv2.circle(img, (90, 105), 6, (20, 20, 20), -1)
        cv2.circle(img, (130, 105), 6, (20, 20, 20), -1)
        cv2.ellipse(img, (110, 145), (20, 14), 0, 0, 180, (20, 20, 20), 2)

        faces = detect_faces(img)

        self.assertGreaterEqual(len(faces), 1)


if __name__ == '__main__':
    unittest.main()
