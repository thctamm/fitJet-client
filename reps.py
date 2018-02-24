import cv2
import numpy as np
import time
from colorSensing import ColorSensing
import peakutils
import requests

cam = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

#cv2.namedWindow("test")
cs = ColorSensing()
img_counter = 0
imgs = []
mdpts = {'x':np.array([]), 'y':np.array([])}
t0 = time.time()
max_pks = 3
min_buffer = 25
exercise_type = 'x'

while time.time() - t0 < 60:
    ret, frame = cam.read()
    if not ret:
        break
    mi, tup = cs.find_bracelet(frame)
    #np.save('img.npy', frame)
    #imgs.append(mi)
    mdpts['x'] = np.append(mdpts['x'], tup[0] + tup[2]/2.0)
    mdpts['y'] = np.append(mdpts['y'], tup[1] + tup[3]/2.0)
    if len(mdpts[exercise_type]) > min_buffer:
        pks = peakutils.indexes(mdpts[exercise_type], thres=0, min_dist=10)
    else:
        print('STILL WAITING FOR DATA - ON {}/{}'.format(len(mdpts[exercise_type]), min_buffer))
        continue
    if len(pks) >= max_pks:
        mdpts[exercise_type] = mdpts[exercise_type][pks[0]:]
        print('REP DETECTED!')
    #print('THIS IS THE TIME TO SAVE MOFO')
    #time.sleep(3)

# np.save('imgs.npy', np.array(imgs))
# np.save('mdpts.npy', np.array(mdpts))

cam.release()

cv2.destroyAllWindows()
