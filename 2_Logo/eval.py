# -*- coding: utf-8 -*-
from typing import Tuple
import cv2
import dlib


# TODO: Импортируйте библиотеки, которые собираетесь использовать

def detect(image, name):
    model_detector = dlib.simple_object_detector(
        f'Detector_logo_{name}.svm')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = model_detector(image)
    bbox = None
    for box in boxes:
        bbox = (box.left(), box.top(), box.right() - box.left(),
                box.bottom() - box.top())
    return name, bbox


def detect_logo(image) -> Tuple[str, Tuple]:
    """
        Функция для детектирования логотипов

        Входные данные: изображение (bgr), прочитано cv2.imread
        Выходные данные: кортеж (Tuple) с названием логотипа и координатами ограничивающей рамки
            (label, (x, y, w, h)),
                где label - строка с названием логотипа;
                x, y - целочисленные координаты левого верхнего угла рамки, ограничивающей логотип;
                w, h - целочисленные ширина и высота рамки, ограничивающей логотип.

        Примечание: Логотип на изображение всегда ровно один!

        Возможные название логотипов:
            cpp, avt, python, altair, kruzhok.

        Примеры вывода:
            ('cpp', (12, 23, 20, 20))

            ('avt', (403, 233, 45, 60))
    """

    # TODO: Отредактируйте эту функцию по своему усмотрению.
    # Для удобства можно создать собственные функции в этом файле.
    # Алгоритм проверки будет вызывать функцию detect_logo, остальные функции должны вызываться из неё.
    label = "avt"
    names = ['cpp', 'python', 'kruzhok', 'altair', 'avt']

    for name in names:

        label, bbox = detect(image, name)
        if bbox:
            break
    print(label, bbox)
    return (label, bbox)
