import cv2
import numpy as np
from colorSensing import ColorSensing
import peakutils
import requests

cam = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

cs = ColorSensing()

mdpts = {
    'R': np.array([]),
    'G': np.array([]),
    'B': np.array([])
}

MAX_PEAKS = 3
MIN_BUFFER = 25
PEAK_OFFSET = 1
COLOR_RESET_WINDOW = 100
PEAK_THRESHOLD = 0
PEAK_MIN_DIST = 10
APP_URL = 'some_random_heroku.com'

ctr = 0
active_colors = None

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # Update the overall ticker, reinitialize update list
    ctr += 1
    saw_reps = []

    if ctr % COLOR_RESET_WINDOW == 1:
        active_colors = requests.get('{}/active_colors'.format(APP_URL))

    # At this point, active colors should be some list like ['R', 'G']
    for color in active_colors:
        _, tup = cs.find_bracelet(frame, color=color)
        mdpts[color] = np.append(mdpts[color], tup[1] + tup[3]/2.0)

        if len(mdpts[color]) > MIN_BUFFER:
            pks = peakutils.indexes(
                mdpts[color],
                thres=PEAK_THRESHOLD,
                min_dist=PEAK_MIN_DIST
            )
        else:
            continue
        if len(pks) >= MAX_PEAKS:
            mdpts[color] = mdpts[color][pks[PEAK_OFFSET]:]
            saw_reps.append(color)

    if saw_reps:
        requests.post(
            '{}/rep_update'.format(APP_URL),
            json=saw_reps
        )

cam.release()

cv2.destroyAllWindows()
