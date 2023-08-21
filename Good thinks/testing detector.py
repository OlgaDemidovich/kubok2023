import cv2
import numpy as np


def IoU(user_box, true_box):
    """IoU = Area of overlap / Area of union
       Output: 0.0 .. 1.0
       Не важно в каком порядке передаются рамки, IoU не изменится.
    """
    x, y, w, h = user_box
    user_box = (x, y, x + w, y + h)

    x, y, w, h = true_box
    true_box = (x, y, x + w, y + h)
    x1 = max(user_box[0], true_box[0])
    y1 = max(user_box[1], true_box[1])
    x2 = min(user_box[2], true_box[2])
    y2 = min(user_box[3], true_box[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = (user_box[2] - user_box[0] + 1) * (
            user_box[3] - user_box[1] + 1)
    box2_area = (true_box[2] - true_box[0] + 1) * (
            true_box[3] - true_box[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


# cap = cv2.VideoCapture(0)

# net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo_best.weights')
net = cv2.dnn.readNetFromDarknet('yolo_people_signs_lights.cfg',
                                 'yolo_people_signs_lights.weights')
yolo_model = cv2.dnn_DetectionModel(net)
yolo_model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

class_names = ["Road works", "Parking", "No entry", "Pedestrian crossing",
               "Movement prohibition", "Artificial roughness", "Give way",
               "Stop", 'Person', 'Lights']

images = ['1.jpg', '32.jpg', '112.jpg', '1016.jpg', '1290.jpg', '1443.jpg', '1447.jpg']

for img in images:
    # status, frame = cap.read()
    frame = cv2.imread(img)
    classes, scores, boxes = yolo_model.detect(frame, nmsThreshold=0.5)
    cl_sc_box = list(zip(classes, scores, boxes))
    for (cl, score, bbox) in cl_sc_box:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, class_names[cl], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) == 27:
        break
