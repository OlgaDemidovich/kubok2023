import cv2
import numpy as np


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


def template(img):
    if abs(img.shape[0] - img.shape[1]) < 3:
        img = cv2.resize(img, (38, 38), None, 0.5, 0.5)
        etalon = cv2.imread('1.png', 0)
        diff = 255 - cv2.absdiff(img, etalon)
        res = []
        for u in diff:
            res.append(sum(u))
        # print('second', sum(res) / (38 * 38 * 255))
        coincidence = sum(res) / (38 * 38 * 255)
        # cv2.putText(frame, str(coincidence), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        return coincidence
    return 0


def detect_defective_parts(video) -> list:
    """
        Функция для детектирования бракованных гаек.

        Входные данные: объект, полученный cv2.VideoCapture, из объекта можно читать кадры методом .read
            На кадрах конвеер, транспортирующий гайки. Гайки перемещаются от нижней границы кадра к верхней.
            Некоторые гайки повреждены: не имеют центрального отверстия, сплющены, разорваны, деформированы.

        Выходные данные: list
            Необходимо вернуть список, состоящий из нулей и единиц, где 0 - гайка надлежащего качества,
                                                                        1 - бракованная гайка.
            Длина списка должна быть равна количеству гаек на видео.

        Примеры вывода:
            [0, 0, 0, 1] - первые 3 гайки целые, последняя - бракованная
            [1, 1, 1] - все 3 гайки бракованные
            [] - на видео не было гаек

    """

    i = 0
    nuts = dict()
    nuts_img = []
    coincidence = dict()
    contours_ = []
    result = []  # пустой список для засенения результата
    while True:  # цикл чтения кадров из видео
        status, frame = video.read()  # читаем кадр
        if not status:  # выходим из цикла, если видео закончилось
            break

        frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        frame = cv2.flip(frame, 0)

        blurred_frame = cv2.GaussianBlur(frame, (7, 7), 8)
        frame_h, frame_w = frame.shape[:2]
        zone_start = int(frame_h * 0.25)
        zone_end = int(frame_h * 0.75)

        thresh = cv2.inRange(blurred_frame, (0, 0, 0), (100, 100, 100))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
            # cv2.rectangle(frame, bbox, (255, 0, 0), 1)
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            # if y2 > zone_start and y < zone_end:
            #     cv2.rectangle(frame, bbox, (0, 255, 0), 1)
            # else:
            #     cv2.rectangle(frame, bbox, (255, 0, 0), 1)
            nut = None
            num_id = None
            for old_num_id, old_bbox in nuts.items():
                if intersects(bbox, old_bbox):
                    num_id = old_num_id
                    nut = thresh[y:y2, x:x2]
                    # cv2.putText(frame, str(nut.shape), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                    contours_.append(cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
                    if num_id not in coincidence:
                        coincidence[num_id] = [template(nut)]
                    else:
                        coincidence[num_id] = coincidence[num_id] + [template(nut)]
                    nuts[num_id] = bbox
                    break

            if num_id is None and y2 > zone_start and y < zone_end:
                num_id = len(nuts)
                nuts[num_id] = bbox
                nuts_img.append(nut)

            if num_id is not None and y >= zone_end:
                nuts[num_id] = (-50, -50, -1, -1)
                coincidence[num_id] = coincidence[num_id] + [template(nut)]
                num_id = None

        # cv2.line(frame, (0, zone_start), (frame_w, zone_start), (0, 0, 255))
        # cv2.line(frame, (0, zone_end), (frame_w, zone_end), (0, 0, 255))
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     quit()
    for value in coincidence.values():
        print(np.array(value).mean())
        result.append(0 if np.array(value).mean() > 0.9 else 1)

    return result
