import os
import unittest

from face_recognition_module import encode_face


class FaceRecognitionModuleTests(unittest.TestCase):
    def test_encode_face_skips_when_deepface_disabled(self):
        os.environ['SKIP_DEEPFACE'] = '1'
        self.assertIsNone(encode_face([[1, 2, 3]]))


if __name__ == '__main__':
    unittest.main()
