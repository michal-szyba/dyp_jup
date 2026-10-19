import cv2
import matplotlib.pyplot as plt
import numpy as np


def _to_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_trial(ax, map_rgb, trial, pad=120, success_px=5.0):
    h, w = map_rgb.shape[:2]
    pts = [trial.gt] + ([trial.est] if trial.found else [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(max(min(xs) - pad, 0)), int(min(max(xs) + pad, w))
    y0, y1 = int(max(min(ys) - pad, 0)), int(min(max(ys) + pad, h))

    ax.imshow(map_rgb[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
    ax.plot(*trial.gt, "+", color="lime", markersize=14, markeredgewidth=2)

    if trial.found:
        color = "lime" if trial.error_px <= success_px else "red"
        ax.plot(*trial.est, "x", color=color, markersize=10, markeredgewidth=2)
        ax.plot([trial.gt[0], trial.est[0]], [trial.gt[1], trial.est[1]], color="yellow", linewidth=1.5)
        ax.set_title(f"kąt {trial.angle:.0f}° | błąd {trial.error_px:.2f} px", fontsize=9)
    else:
        ax.set_title(f"kąt {trial.angle:.0f}° | NIE ZNALEZIONO", fontsize=9)
    ax.axis("off")


def plot_mosaic(map_img, trials, rows=3, cols=3, success_px=5.0):
    map_rgb = _to_rgb(map_img)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = np.atleast_1d(axes).flatten()
    for ax, trial in zip(axes, trials):
        plot_trial(ax, map_rgb, trial, success_px=success_px)
    for ax in axes[len(trials):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_error_hist(trials, success_px=5.0, title=""):
    errs = np.array([t.error_px for t in trials if t.found])
    fig, ax = plt.subplots(figsize=(9, 4))
    if errs.size == 0:
        ax.text(0.5, 0.5, "brak dopasowań", ha="center", va="center", transform=ax.transAxes)
        return fig
    errs = np.clip(errs, 1e-3, None)
    bins = np.logspace(np.log10(errs.min()) - 0.05, np.log10(errs.max()) + 0.05, 30)
    ax.hist(errs, bins=bins, color="crimson", edgecolor="black", alpha=0.7)
    ax.axvline(success_px, color="black", linestyle="--", label=f"próg sukcesu = {success_px:g} px")
    ax.set_xscale("log")
    ax.set_xlabel("Błąd euklidesowy [px]")
    ax.set_ylabel("Liczba prób")
    ax.set_title(title or f"Rozkład błędu lokalizacji ({len(trials)} prób, znaleziono {errs.size})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_accuracy_pie(trials, edges=(0.2, 0.5, 5.0)):
    labels = ["Niedopasowane"]
    counts = [sum(not t.found for t in trials)]
    lower = 0.0
    for e in edges:
        labels.append(f"{lower:g}–{e:g} px" if lower else f"< {e:g} px")
        counts.append(sum(t.found and lower <= t.error_px < e for t in trials))
        lower = e
    labels.append(f"> {edges[-1]:g} px (fałszywe)")
    counts.append(sum(t.found and t.error_px >= edges[-1] for t in trials))

    colors = ["#7f8c8d", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"][: len(labels)]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 6))
    _, _, autotexts = ax.pie(
        counts, labels=labels, colors=colors, startangle=140,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})" if p > 0 else "",
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(f"Struktura dokładności lokalizacji UAV ({total} prób)", fontsize=13, pad=20)
    fig.tight_layout()
    return fig


def plot_error_map(map_img, trials, success_px=5.0, max_dim=2000):
    h, w = map_img.shape[:2]
    vis_scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(_to_rgb(map_img), (int(w * vis_scale), int(h * vis_scale)),
                       interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(small)
    for t in trials:
        gx, gy = t.gt[0] * vis_scale, t.gt[1] * vis_scale
        if t.found:
            ex, ey = t.est[0] * vis_scale, t.est[1] * vis_scale
            color = "#2ecc71" if t.error_px <= success_px else "#e74c3c"
            ax.plot([gx, ex], [gy, ey], color=color, linewidth=1.2, alpha=0.75)
            ax.plot(gx, gy, "o", color="#2ecc71", markersize=4)
            ax.plot(ex, ey, "x", color=color, markersize=6, markeredgewidth=1.5)
        else:
            ax.plot(gx, gy, "o", color="black", markeredgecolor="#e74c3c", markersize=5, markeredgewidth=1.5)

    ax.plot([], [], "o", color="#2ecc71", linestyle="None", label="Prawda terenowa (GT)")
    ax.plot([], [], "x", color="#2ecc71", linestyle="None", label=f"Trafienie (błąd ≤ {success_px:g} px)")
    ax.plot([], [], "x", color="#e74c3c", linestyle="None", label=f"Błędna transformacja (> {success_px:g} px)")
    ax.plot([], [], "o", color="black", markeredgecolor="#e74c3c", linestyle="None", label="Brak dopasowania")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_title(f"Przestrzenny rozkład błędów ({len(trials)} prób)", fontsize=13, pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig