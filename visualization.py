import cv2
from matplotlib import pyplot as plt


def visualize_comparison(map_image, cutout_image, actual, detected):
    h, w = cutout_image.shape[:2]
    output_map = map_image.copy()

    cv2.rectangle(output_map, (actual[0], actual[1]),
                  (actual[0] + w, actual[1] + h), (0, 255, 0), 8)

    offset = 5
    cv2.rectangle(output_map, (detected[0] + offset, detected[1] + offset),
                  (detected[0] + w - offset, detected[1] + h - offset), (0, 0, 255), 3)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output_map, cv2.COLOR_BGR2RGB))
    plt.title("Green = Actual, Red = Detected (Inset)")
    plt.axis('off')
    plt.show()