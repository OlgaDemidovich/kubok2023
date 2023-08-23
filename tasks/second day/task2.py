import os
import time

import cv2
import numpy as np
import pigpio


# 192.168.4.1
# raspberry
def get_angle():
    global angle, last, perspective
    img = cv2.resize(frame, SIZE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.inRange(gray, 195, 255)

    matrix_trans = cv2.getPerspectiveTransform(TRAP, RECT)
    perspective = cv2.warpPerspective(binary, matrix_trans, SIZE,
                                      flags=cv2.INTER_LINEAR)

    hist = np.sum(perspective, axis=0)
    mid = hist.shape[0] // 2
    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid

    center = (left + right) // 2

    err = 0 - (center - SIZE[0] // 2)
    # print('Err:', err)
    # angle = int(90 + KP * err)
    angle = int(90 + KP * err + KD * (err - last))
    # print('Angle:', angle)
    last = err

    angle = max([angle, 72])
    angle = min([angle, 108])


def detect_light(frame_light):
    gray = cv2.cvtColor(frame_light, cv2.COLOR_BGR2GRAY)
    binary = cv2.inRange(gray, 240, 255)
    frame_res = cv2.resize(binary, (60, 60))
    # frame_res = frame_res[:, 10:50]
    # cv2.imshow('fr', frame_res)
    # print(frame_res)
    red = np.sum(frame_res[5:15, :])
    yellow = np.sum(frame_res[25:35, :])
    green = np.sum(frame_res[45:55, :])
    # print(red, yellow, green)
    print(green)
    if green > 2000:
        return 'green'
    elif red > 60000 and yellow > 100000:
        return 'red and yellow'
    elif red > 60000:
        return 'red'
    elif yellow > 100000:
        return 'yellow'
    return 'none'


def setup_gpio():
    os.system('sudo pigpiod')
    time.sleep(1)
    ESC = 17
    STEER = 18
    pi = pigpio.pi()
    pi.set_servo_pulsewidth(ESC, 0)
    pi.set_servo_pulsewidth(STEER, 0)
    time.sleep(1)
    return (pi, ESC, STEER)


def control(speed, angle):
    pi.set_servo_pulsewidth(ESC, speed)
    pi.set_servo_pulsewidth(STEER, int(11.1 * angle + 500))


SIZE = (533, 300)

KP = 0.32
KD = 0.17

RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])

src_draw = np.array(TRAP, dtype=np.int32)
last = 0
color = 'none'
previous_color = 'none'

pi, ESC, STEER = setup_gpio()
angle = 90

control(1500, 90)
time.sleep(1.5)

cap = cv2.VideoCapture(0)

perspective = np.array([])

straight = False
t0 = 0
t1 = 0
count = 0
count_green = 0
stop = False
net = cv2.dnn.readNetFromDarknet('yolov4-tiny-obj.cfg', 'yolov4-tiny-obj_best.weights')
yolo_model = cv2.dnn_DetectionModel(net)
yolo_model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
models = yolo_model

while True:
    status, frame = cap.read()
    if not stop:
        if straight:
            angle = 90
        # if count == 1 and time.monotonic() - t1 > 2:
        #     get_angle()
        else:
            get_angle()
        control(1557, angle)
        if not straight and np.sum(perspective[120:190, :]) >= 5400000:
            straight = True
            stop = True
            continue
        if straight and time.monotonic() - t0 >= 2.4:
            straight = False
            count += 1
        if count == 1:
            t1 = time.monotonic()
            break


    else:
        control(1500, 90)
        classes, scores, boxes = yolo_model.detect(frame, 0.45, 0.25)

        for bbox in boxes:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = detect_light(frame[y:y + h, x:x + w])
            print(color, count_green)
            if color == 'green':
                count_green += 1
            else:
                count_green = 0
            if count_green >= 2:
                stop = False
                straight = True
                t0 = time.monotonic()

control(1500, 90)

cap.release()
