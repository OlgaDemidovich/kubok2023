import cv2
import pandas as pd
import csv
import os

# for i in range(1, 7):
#     name = f'{i}.png'
#     frame = cv2.imread('reference/'+name)
#     # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
#
#     blurred_frame = cv2.GaussianBlur(frame, (7, 7), 8)
#
#     thresh = cv2.inRange(blurred_frame, (0, 0, 0), (100, 100, 100))
#
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     for cnt in contours:
#         bbox = cv2.boundingRect(cnt)
#         # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
#         # cv2.rectangle(frame, bbox, (255, 0, 0), 1)
#         x, y, w, h = bbox
#         x2, y2 = x + w, y + h
#         nut = thresh[y:y2, x:x2]
#     cv2.imwrite(f'reference/_{name}', nut)

def intersects(bbox1, bbox2):
    x, y, w, h = bbox1
    bbox1 = (x, y, x + w, y + h)
    x, y, w, h = bbox2
    bbox2 = (x, y, x + w, y + h)
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    return inter_area > 0


def detect_defective_parts(video):
    i = 0
    nuts = dict()
    nuts_img = []
    coincidence = dict()
    result = []  # пустой список для засенения результата
    while True:  # цикл чтения кадров из видео
        status, frame = video.read()  # читаем кадр
        if not status:  # выходим из цикла, если видео закончилось
            break

        # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        frame = cv2.flip(frame, 0)

        blurred_frame = cv2.GaussianBlur(frame, (7, 7), 8)
        frame_h, frame_w = frame.shape[:2]
        zone_start = int(frame_h * 0.25)
        zone_end = int(frame_h * 0.75)

        thresh = cv2.inRange(blurred_frame, (0, 0, 0), (100, 100, 100))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            nut = None
            num_id = None
            for old_num_id, old_bbox in nuts.items():
                if intersects(bbox, old_bbox):
                    num_id = old_num_id
                    nut = frame[y:y2, x:x2]
                    class_name = real_label[num_id]
                    name = len([n for n in os.listdir('images/')])
                    cv2.imwrite(f'images/{name}.jpg', nut)
                    file_writer.writerow([f'images/{name}.jpg', class_name])
                    nuts[num_id] = bbox
                    break

            if num_id is None and y2 > zone_start and y < zone_end:
                num_id = len(nuts)
                nuts[num_id] = bbox
                nuts_img.append(nut)

            if num_id is not None and y >= zone_end:
                nuts[num_id] = (-50, -50, -1, -1)


with open("annotations_nuts.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    file_writer.writerow(["image", "class"])

    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=',', dtype=str)
    data = data.sample(frac=1)

    for i, row in enumerate(data.itertuples()):
        row_id, video_filename, real_label = row

        print(video_filename)
        # convert string of 0 and 1 to list
        real_label = list(map(int, list(real_label)))

        video = cv2.VideoCapture(video_filename)
        detect_defective_parts(video)
        video.release()
