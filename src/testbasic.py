import cv2
import numpy as np
import pytest

from src.analysis import intersection_over_union, run_experiment, summarize
from src.uav import MapLocalizer, cutout_rotate


def blob_map(size, center, sigma=8.0):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    img = 255.0 * np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (2 * sigma ** 2))
    return img.astype(np.float32)


@pytest.mark.parametrize("angle", [0, 37, 90, 181, 299])
@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("center", [(400.0, 500.0), (400.4, 500.7)])
def test_ground_truth_is_crop_center(angle, scale, center):
    img = blob_map(900, center)
    crop, gt = cutout_rotate(img, angle, scale, 200, 200, center=center)
    assert gt == center
    ys, xs = np.mgrid[0:200, 0:200]
    cx = (xs * crop).sum() / crop.sum()
    cy = (ys * crop).sum() / crop.sum()
    assert np.hypot(cx - 99.5, cy - 99.5) < 0.3


def test_iou():
    assert intersection_over_union((0, 0, 10, 20), (0, 0, 10, 20)) == pytest.approx(1.0)
    assert intersection_over_union((0, 0, 10, 20), (0, 0, 20, 10)) == pytest.approx(1 / 3)
    assert intersection_over_union((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    assert intersection_over_union(None, (0, 0, 1, 1)) == 0.0


def synthetic_map(size=1500, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(size, size)).astype(np.float32)
    img = sum(cv2.GaussianBlur(base, (0, 0), s) * s for s in (2, 4, 8, 16))
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("preprocess", ["gray", "sobel"])
def test_localizer_end_to_end(preprocess):
    map_img = synthetic_map()
    localizer = MapLocalizer(map_img, preprocess=preprocess)
    trials = run_experiment(map_img, localizer, n_trials=12, seed=3, scale_range=(0.9, 1.1))
    s = summarize(trials, success_px=2.0)
    assert s["success_rate"] >= 75


def test_same_seed_same_trials():
    map_img = synthetic_map()
    localizer = MapLocalizer(map_img)
    a = run_experiment(map_img, localizer, n_trials=3, seed=5)
    b = run_experiment(map_img, localizer, n_trials=3, seed=5)
    assert [t.gt for t in a] == [t.gt for t in b]