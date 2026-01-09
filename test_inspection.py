import unittest, os, cv2, numpy as np
from inspection import PCBAligner, PCBDetector
from data_prep import convert_box

class TestPCBAligner(unittest.TestCase):
    def test_align_identical(self):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        aligner = PCBAligner()
        aligned = aligner.align(img, img)
        self.assertEqual(aligned.shape, img.shape)
    
    def test_align_rotated(self):
        img = cv2.imread('data/DeepPCB/PCBData/group00041/00041/00041_temp.jpg') if os.path.exists('data/DeepPCB/PCBData/group00041/00041/00041_temp.jpg') else np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        aligner = PCBAligner()
        aligned = aligner.align(img, rotated)
        self.assertEqual(aligned.shape, img.shape)

class TestConversion(unittest.TestCase):
    def test_convert_box(self):
        box = (100, 100, 200, 200)
        result = convert_box((640, 640), box)
        self.assertAlmostEqual(result[0], 0.234375, places=4)
        self.assertAlmostEqual(result[1], 0.234375, places=4)
        self.assertAlmostEqual(result[2], 0.15625, places=4)
        self.assertAlmostEqual(result[3], 0.15625, places=4)

if __name__ == '__main__': unittest.main()
