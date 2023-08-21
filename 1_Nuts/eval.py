import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("MultiClas_Conv_v6.h5")


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


def template(img, img_thresh):
    # img = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    image1 = cv2.resize(img, (32, 32)) / 255
    image1 = np.expand_dims(image1, axis=0)
    pred = model.predict(image1)
    res = 1 - pred[0, 0] + pred[0, 1]
    # res = 1 if pred[0, 1] > pred[0, 0] else 0
    # print(res)

    img = cv2.resize(img_thresh, (138, 138), None, 0.5, 0.5)
    etalon = cv2.imread('4.png', 0)
    etalon = cv2.resize(etalon, (138, 138), None, 0.5, 0.5)
    diff = 255 - cv2.absdiff(img, etalon)
    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    result = []
    for u in diff:
        result.append(sum(u))
    # print('second', sum(res) / (38 * 38 * 255))
    coincidence = sum(result) / (138 * 138 * 255)
    # cv2.putText(frame, str(coincidence), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    res += (coincidence - 0.1) * 9
    # print(res/2)
    # res = 1 if res / 2 > 0.5 else 0
    return res


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
                    nut = frame[y:y2, x:x2]
                    nut_thresh = thresh[y:y2, x:x2]
                    # cv2.putText(frame, str(nut.shape), (x, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                    # if num_id not in coincidence:
                    #     coincidence[num_id] = [template(nut, model, nut_thresh)]
                    # else:
                    #     coincidence[num_id] = coincidence[num_id] + [
                    #         template(nut, model, nut_thresh)]
                    nuts[num_id] = bbox
                    break
            if num_id is None and y2 > zone_start and y < zone_end:
                num_id = len(nuts)
                nuts[num_id] = bbox

            if num_id is not None and y >= zone_end:
                nuts[num_id] = (-50, -50, -1, -1)
                result.append(template(nut, nut_thresh))

        # cv2.line(frame, (0, zone_start), (frame_w, zone_start), (0, 0, 255))
        # cv2.line(frame, (0, zone_end), (frame_w, zone_end), (0, 0, 255))
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     quit()

    # print(coincidence)
    # for value in coincidence.values():
    #     m = np.array(value).mean()
    #     print(m)
    #     result.append(1 if m > 0.91 else 0)
    print([i / 2 for i in result])
    result = [1 if i / 2 > 0.7 else 0 for i in result]
    print(result)
    return result
