from dataclasses import dataclass

import cv2
import numpy as np

from .filter import PREPROCESSORS, to_gray


def cutout_rotate(img, angle, scale=1.0, target_w=200, target_h=200, center=None, rng=None):

    h_map, w_map = img.shape[:2]

    margin = int(np.ceil(np.hypot(target_w, target_h) / (2.0 * scale))) + 2
    if w_map - 1 <= 2 * margin or h_map - 1 <= 2 * margin:
        raise ValueError("Mapa jest za mała, aby bezpiecznie wyciąć taki kadr.")

    if center is None:
        rng = np.random.default_rng() if rng is None else rng
        cx = rng.uniform(margin, w_map - 1 - margin)
        cy = rng.uniform(margin, h_map - 1 - margin)
    else:
        cx, cy = float(center[0]), float(center[1])
        if not (margin <= cx <= w_map - 1 - margin and margin <= cy <= h_map - 1 - margin):
            raise ValueError("Środek kadru jest zbyt blisko krawędzi mapy.")

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += (target_w - 1) / 2.0 - cx
    M[1, 2] += (target_h - 1) / 2.0 - cy
    uav = cv2.warpAffine(img, M, (target_w, target_h),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return uav, (cx, cy)


def locate_template(map_img, uav_img):
    map_gray = to_gray(map_img)
    uav_gray = to_gray(uav_img)
    result = cv2.matchTemplate(map_gray, uav_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    h, w = uav_gray.shape[:2]
    return (x + (w - 1) / 2.0, y + (h - 1) / 2.0), float(score)


@dataclass
class Localization:
    center: tuple | None
    inliers: int = 0
    n_matches: int = 0
    transform: np.ndarray | None = None
    reason: str = ""


class MapLocalizer:
    def __init__(self, map_img, preprocess="gray", ratio=0.75, min_inliers=10,
                 ransac_thresh=3.0, scale_range=(0.4, 2.5)):
        self.preprocess = PREPROCESSORS[preprocess] if isinstance(preprocess, str) else preprocess
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.scale_range = scale_range

        self.sift = cv2.SIFT_create()
        self.kp_map, self.des_map = self.sift.detectAndCompute(self.preprocess(map_img), None)
        if self.des_map is None:
            raise ValueError("Nie znaleziono punktów charakterystycznych na mapie.")

        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.matcher.add([self.des_map])
        self.matcher.train()

    def _fail(self, reason, n_matches=0):
        return Localization(None, n_matches=n_matches, reason=reason)

    def locate(self, uav_img):
        gray = self.preprocess(uav_img)
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 2:
            return self._fail("za mało punktów charakterystycznych na kadrze")

        pairs = self.matcher.knnMatch(des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < self.min_inliers:
            return self._fail("za mało dopasowań po teście Lowe'a", len(good))

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        A, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000, confidence=0.995)
        M = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])


        if M is None or mask is None:
            return self._fail("nie udało się wyznaczyć transformacji", len(good))

        inliers = int(mask.sum())
        if inliers < self.min_inliers:
            return self._fail("za mało inlierów po RANSAC", len(good))

        scale = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
        if not (self.scale_range[0] <= scale <= self.scale_range[1]):
            return self._fail("nierealna skala transformacji", len(good))

        h, w = gray.shape[:2]
        center_uav = np.array([[[(w - 1) / 2.0, (h - 1) / 2.0]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(center_uav, M)[0, 0]
        if not (np.isfinite(x) and np.isfinite(y)):
            return self._fail("zdegenerowana transformacja", len(good))

        return Localization((float(x), float(y)), inliers, len(good), M)