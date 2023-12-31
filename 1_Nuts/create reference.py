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


def save_file(path, class_name, image):
    cv2.imwrite(path, image)
    file_writer.writerow([path, class_name])
    img = [path, image]

    path = path.replace('.jpg', '')

    horPath = path + "_horflip.jpg"
    horflip = cv2.flip(image, 1)
    cv2.imwrite(horPath, horflip)
    file_writer.writerow([horPath, class_name])
    hor = [horPath, horflip]

    verPath = path + "_verflip.jpg"
    verflip = cv2.flip(image, 0)
    cv2.imwrite(verPath, verflip)
    file_writer.writerow([verPath, class_name])
    ver = [verPath, verflip]

    invPath = path + "_invflip.jpg"
    invflip = cv2.flip(image, -1)
    cv2.imwrite(invPath, invflip)
    file_writer.writerow([invPath, class_name])
    inv = [invPath, invflip]

    # for dark_path, dark_image in [img, hor, ver, inv]:
    #     darkPath = dark_path + "_dark.jpg"
    #     darkImage = dark_image // 2
    #     cv2.imwrite(darkPath, darkImage)
    #     file_writer.writerow([darkPath, class_name])


def detect_defective_parts(video):
    i = 0
    nuts = dict()
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

            # nut = frame[y:y2, x:x2]
            nut = thresh[y:y2, x:x2]
            num_id = None
            name = len(os.listdir('images/')) // 4
            path = f'images/{name}.jpg'
            for old_num_id, old_bbox in nuts.items():
                if intersects(bbox, old_bbox):
                    num_id = old_num_id
                    nuts[num_id] = bbox
                    class_name = real_label[num_id]

                    break

            if num_id is None and y2 > zone_start and y < zone_end:
                num_id = len(nuts)
                class_name = real_label[num_id]
                nuts[num_id] = bbox
                save_file(path, class_name, nut)

            if y2 == int(frame_h * 0.5):
                save_file(path, class_name, nut)

            if num_id is not None and y >= zone_end:
                nuts[num_id] = (-50, -50, -1, -1)
                save_file(path, class_name, nut)


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
