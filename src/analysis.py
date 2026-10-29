from dataclasses import dataclass

import numpy as np

from .filter import _sample, apply_uav_effects
from .uav import cutout_rotate


def intersection_over_union(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter_area == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter_area
    return inter_area / float(union)


@dataclass
class Trial:
    angle: float
    scale: float
    gt: tuple
    est: tuple | None
    inliers: int
    error_px: float

    @property
    def found(self):
        return self.est is not None


def run_experiment(uav_img, localizer, n_trials=100, seed=0, crop=(200, 200),
                   angle_range=(0.0, 360.0), scale_range=(1.0, 1.0), effects=None):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        angle = rng.uniform(*angle_range)
        scale = rng.uniform(*scale_range)
        uav, gt = cutout_rotate(uav_img, angle, scale, crop[0], crop[1], rng=rng)
        if effects is not None:
            uav = apply_uav_effects(uav, rng=rng, **effects)

        loc = localizer.locate(uav)
        if loc.center is None:
            err = np.nan
        else:
            err = float(np.hypot(loc.center[0] - gt[0], loc.center[1] - gt[1]))
        trials.append(Trial(angle, scale, gt, loc.center, loc.inliers, err))
    return trials


def summarize(trials, success_px=5.0):
    n = len(trials)
    errs = np.array([t.error_px for t in trials], dtype=float)
    found = ~np.isnan(errs)
    ok = found & (errs <= success_px)
    ok_errs = errs[ok]
    found_errs = errs[found]

    def stat(values, fn):
        return float(fn(values)) if values.size else float("nan")

    return {
        "n": n,
        "found_rate": found.sum() / n * 100,
        "success_rate": ok.sum() / n * 100,
        "false_match_rate": (found & ~ok).sum() / n * 100,

        "median_px": stat(ok_errs, np.median),
        "p90_px": stat(ok_errs, lambda e: np.percentile(e, 90)),
        "p95_px": stat(ok_errs, lambda e: np.percentile(e, 95)),
        "rmse_px": stat(ok_errs, lambda e: np.sqrt(np.mean(e ** 2))),

        "median_all_px": stat(found_errs, np.median),
        "p90_all_px": stat(found_errs, lambda e: np.percentile(e, 90)),
        "rmse_all_px": stat(found_errs, lambda e: np.sqrt(np.mean(e ** 2))),

        "success_px": success_px,
    }


def print_summary(name, s):
    print(f"[{name}] prób: {s['n']}")
    print(f"  znaleziono:            {s['found_rate']:.1f}%")
    print(f"  sukces (<= {s['success_px']:g} px):   {s['success_rate']:.1f}%")
    print(f"  fałszywe dopasowania:  {s['false_match_rate']:.1f}%")
    print(f"  błąd dla sukcesów: mediana {s['median_px']:.2f} px, "
          f"p90 {s['p90_px']:.2f} px, p95 {s['p95_px']:.2f} px, RMSE {s['rmse_px']:.2f} px")