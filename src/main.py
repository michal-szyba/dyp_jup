import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from src import visualization as viz
from src.analysis import print_summary, run_experiment, summarize
from src.uav import MapLocalizer

MAP_PATH = "../data/slichowice.png"
N_TRIALS = 100
SEED = 42
SUCCESS_PX = 5.0


def main():
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        raise FileNotFoundError(f"Nie można wczytać mapy: {MAP_PATH}")

    effects = dict(brightness=(10, 30), contrast=(0.9, 1.3), noise_level=(10, 25))
    results = {}
    for name in ("gray", "sobel"):
        localizer = MapLocalizer(map_img, preprocess=name)
        trials = run_experiment(map_img, localizer, n_trials=N_TRIALS, seed=SEED,
                                scale_range=(0.8, 1.2), effects=effects)
        print_summary(name, summarize(trials, SUCCESS_PX))
        results[name] = trials

    for name, trials in results.items():
        viz.plot_error_hist(trials, SUCCESS_PX, title=f"Rozkład błędu - {name}")
        viz.plot_accuracy_pie(trials)
    viz.plot_mosaic(map_img, results["gray"], success_px=SUCCESS_PX)
    viz.plot_error_map(map_img, results["gray"], SUCCESS_PX)
    plt.savefig('result.png')


if __name__ == "__main__":
    main()