import random

import cv2


def cutout(img):
    h, w = img.shape[:2]
    if w < 128 or h < 128:
        raise ValueError('Image too small')
    max_x = w - 128
    max_y = h - 128
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    cutout = img[y:y + 128, x:x + 128]
    return cutout, (x, y)

def locate(map, uav):
    map_gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    uav_gray = cv2.cvtColor(uav, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(map_gray, uav_gray, cv2.TM_CCOEFF_NORMED)
    _, value, _, location = cv2.minMaxLoc(result)
    return location, value

