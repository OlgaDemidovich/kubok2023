import cv2

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo_best.weights')
yolo_model = cv2.dnn_DetectionModel(net)
yolo_model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
models = [yolo_model]
class_names = ["Road works", "Parking", "No entry", "Pedestrian crossing",
               "Movement prohibition", "Artificial roughness", "Give way",
               "Stop", 'Person', 'Lights']

while True:
    status, frame = cap.read()
    classes, scores, boxes = yolo_model.detect(frame)
    print(classes)
    for cl, bbox in zip(classes, boxes):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, class_names[cl], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
    print(classes)
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break
