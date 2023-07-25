# -*- coding: utf-8 -*-
"""
Файл служит для определения точности вашего алгоритма

Для получения оценки точности, запустите файл на исполнение
"""

import eval as submission
import cv2
import pandas as pd


DETECTION_THRESHOLD = 0.5


def IoU(user_box, true_box):
    """IoU = Area of overlap / Area of union
       Output: 0.0 .. 1.0
       Не важно в каком порядке передаются рамки, IoU не изменится.
    """
    x, y, w, h = user_box
    user_box = (x, y, x+w, y+h)

    x, y, w, h = true_box
    true_box = (x, y, x+w, y+h)

    x1 = max(user_box[0], true_box[0])
    y1 = max(user_box[1], true_box[1])
    x2 = min(user_box[2], true_box[2])
    y2 = min(user_box[3], true_box[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = (user_box[2] - user_box[0] + 1) * (user_box[3] - user_box[1] + 1)
    box2_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def check_answer(real_answer, user_answer):
    real_cls, real_bbox = real_answer
    user_cls, user_bbox = user_answer

    if real_cls != user_cls:
        print(f"Wrong class! Expected '{real_cls}', but got '{user_cls}'")
        return False
    
    if IoU(real_bbox, user_bbox) < DETECTION_THRESHOLD:
        print('Low IoU:', IoU(real_bbox, user_bbox))
        return False

    return True


def main():
    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=',')
    data = data.sample(frac=1)

    correct = 0
    for i, row in enumerate(data.itertuples()):
        row_id, image_filename, real_label, x, y, w, h = row
        real_bbox = (x, y, w, h)

        real_answer = (real_label, real_bbox)
        image = cv2.imread(image_filename)

        user_answer = submission.detect_logo(image)

        if check_answer(real_answer, user_answer):
            correct += 1
            print(image_filename, '- верно')
        else:
            print(image_filename, '- неверно')

    total_object = len(data.index)
    print(f"Из {total_object} логотипов верно определены {correct}")

    score = correct / total_object
    print(f"Точность: {score:.2f}")


if __name__ == '__main__':
    main()
