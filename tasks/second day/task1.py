# -*- coding: utf-8 -*-
from typing import Tuple

import cv2
import numpy as np
# TODO: Допишите импорт библиотек, которые собираетесь использовать






net = cv2.dnn.readNetFromDarknet('yolov4-tiny-obj.cfg', 'yolov4-tiny-obj_best.weights')
yolo_model = cv2.dnn_DetectionModel(net)
yolo_model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
models = [yolo_model]
def detect_light(img):

cam = cv2.VideoCapture(0)
while True:
    status, image = cam.read()
    yolo_model = models[0]
    classes, scores, boxes = yolo_model.detect(image, 0.45, 0.25)
    bbox = boxes[0]
    box = []
    a = int(bbox[0])
    box.append(a)
    b = int(bbox[1])
    box.append(b)
    c = int(bbox[2])
    box.append(c)
    d = int(bbox[3])
    box.append(d)
    print(bbox)
    x, y, w, h = bbox
    copy = image.copy()
    cv2.rectangle(copy, bbox, (255, 0, 0), 3)
    image = image[y : y+h, x : x+w]
    cv2.imshow('i', copy)
    cv2.waitKey(10)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.inRange(gray, 240, 255)
    frame_res = cv2.resize(binary, (60, 60))
    # frame_res = frame_res[:, 10:50]
    # cv2.imshow('fr', frame_res)
    # print(frame_res)
    red = np.sum(frame_res[5:15, :])
    yellow = np.sum(frame_res[25:35, :])
    green = np.sum(frame_res[45:55, :])
    print(red, yellow, green)
    ind = np.argmax([red, yellow, green])
    if green > 2000:
        print ('green')
    elif red > 60000 and yellow > 90000:
        print('red yellow')
    elif red>60000:
        print ('red')
    elif yellow > 90000:
        print ('yellow')
    else:
        print("none")
    #print(bbox)
