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
    cv2.imshow('fr', frame_res)
    print(frame_res)
    red = np.sum(frame_res[5:15, :])
    yellow = np.sum(frame_res[25:35, :])
    green = np.sum(frame_res[45:55, :])
    print(red, yellow, green)
    ind = np.argmax([red, yellow, green])
    if ind == 2:
        return 'green'
    elif ind == 0:
        return 'red'
    elif ind == 1:
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

pi, ESC, STEER = setup_gpio()
angle = 90

control(1500, 90)
time.sleep(1.5)

cap = cv2.VideoCapture(0)

perspective = np.array([])

flag = False
t0 = 0
count = 0
# net = cv2.dnn.readNetFromDarknet('yolo_people_signs_lights.cfg',
#                                  'yolo_people_signs_lights.weights')
# yolo_model = cv2.dnn_DetectionModel(net)
# yolo_model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
# models = yolo_model
# class_names = ["Road works", "Parking", "No entry", "Pedestrian crossing",
#                "Movement prohibition", "Artificial roughness", "Give way",
#                "Stop", 'Person', 'Lights']

while True:
    status, frame = cap.read()
    # classes, scores, boxes = yolo_model.detect(frame, nmsThreshold=0.5)
    # for (cl, score, bbox) in list(zip(classes, scores, boxes)):
    #     if class_names[cl] == 'Lights':
    #         x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    #         color = detect_light(frame[y:y + h, x:x + w])
    #         print(color)
    #         if color == 'green':
    #             status = True

    # print(class_names[classes[0]])
    if status:
        if flag:
            angle = 84
        else:
            get_angle()
        print(angle)
        control(1558, angle)
        if time.monotonic() - t0 >= 1.7 and not flag and np.sum(perspective[130:200, :]) >= 5400000:
            print('-------------detect stop')
            control(1558, 84)
            flag = True
            t0 = time.monotonic()
        if flag and time.monotonic() - t0 >= 1.7:
            flag = False
            count += 1
        if count == 2:
            break

control(1500, 90)

cap.release()
