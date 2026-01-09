import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class PCBAligner:
    def __init__(self, n_feat=5000):
        self.sift, self.matcher = cv2.SIFT_create(n_features=n_feat), cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def align(self, temp, tgt):
        kp1, des1 = self.sift.detectAndCompute(temp, None)
        kp2, des2 = self.sift.detectAndCompute(tgt, None)
        good = [m for m, n in self.matcher.knnMatch(des1, des2, k=2) if m.distance < 0.7 * n.distance]
        if len(good) < 4: raise ValueError("Insufficient matches")
        src, dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2), np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return cv2.warpPerspective(tgt, cv2.findHomography(dst, src, cv2.RANSAC, 5.0)[0], (temp.shape[1], temp.shape[0]))

class PCBDetector:
    def __init__(self, path, conf=0.25):
        self.model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=path, confidence_threshold=conf, device='cuda:0')

    def detect(self, img):
        return get_sliced_prediction(img if not isinstance(img, str) else cv2.imread(img), self.model, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
