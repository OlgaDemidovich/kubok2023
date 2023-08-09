import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import yolopy

from arduino import Arduino
from utils import *


def right_drive():
    arduino.set_angle(angle)


def get_angle():
    global last_err

    left, right = centre_mass(perspective)
    print(left, right)
    err = 0 - ((left + right) // 2 - SIZE[0] // 2)
    if abs(right - left) < 100:
        err = last_err

    angle = int(90 + KP * err + KD * (err - last_err))

    if angle < 60:
        angle = 60
    elif angle > 120:
        angle = 120

    last_err = err
    # print(f'angle={angle}')
    return angle


def stoped():
    global last_err

    left, right = centre_mass(perspective)

    err = 0 - ((left - 20 + 300) // 2 - SIZE[0] // 2)
    if abs(300 - left + 20) < 100:
        err = last_err

    angle = int(90 + KP * err + KD * (err - last_err))

    if angle < 60:
        angle = 60
    elif angle > 120:
        angle = 120

    last_err = err
    # print(f'angle={angle}')
    return angle


CAR_SPEED = 1430
ARDUINO_PORT = '/dev/ttyUSB0'
CAMERA_ID = '/dev/video0'

KP = 0.55  # 0.22 0.32 0.42
KD = 0.25  # 0.17
last = 0

SIZE = (533, 300)

RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])

src_draw = np.array(TRAP, dtype=np.int32)

# OPENCV PARAMS
THRESHOLD = 200
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
model_file = 'yolov4-tiny.tmfile'
model = yolopy.Model(model_file, use_uint8=True, use_timvx=True, cls_num=10)
class_names = ["Road works", "Parking", "No entry", "Pedestrian crossing",
                   "Movement prohibition", "Artificial roughness", "Give way",
                   "Stop", 'Person', 'Lights']

arduino = Arduino(ARDUINO_PORT, baudrate=115200, timeout=10)
time.sleep(2)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

arduino.set_speed(CAR_SPEED)

last_err = 0
stop_line = False
flag_stop = False
detect_parking = False
t_0 = 0
ind_step = 0
start_position = '2'

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, SIZE)

        binary = binarize(img, THRESHOLD)
        perspective = trans_perspective(binary, TRAP, RECT, SIZE)
        angle = get_angle()        

        if not detect_parking:
            classes, scores, boxes = model.detect(frame[100:, :])
            detected = [class_names[cls] for cls in classes]
            if 'Parking' in detected:
                print('---------------------detected')
                t_0 = time.monotonic()
                detect_parking = True
                arduino.set_speed(1440)
        if detect_parking and time.monotonic() - t_0 <= 1.3:
            angle = 60
        if detect_parking and 1.3 <= time.monotonic() - t_0 <= 2.8:
            angle = 120
        if detect_parking and 2.8 <= time.monotonic() - t_0 <= 3:
            angle = stoped()
        if detect_parking and time.monotonic() - t_0 >= 4:
            break
        arduino.set_angle(angle)
        
except KeyboardInterrupt as e:
    print('Program stopped!', e)

arduino.stop()
arduino.set_angle(90)
cap.release()
