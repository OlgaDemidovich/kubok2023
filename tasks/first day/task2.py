import os
import time

import cv2
import numpy as np
import pigpio


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


def get_path(start, end):
    point = start
    path = [point]
    count = 0
    while point != end:
        point = input_graph[point]
        if str(path[-1]) + str(point) in ind_per:
            count += 1
        path.append(point)
    return count, path


def get_start_point_coordinates(img):
    red = cv2.inRange(img, (0, 0, 107), (255, 104, 237))
    x = np.nonzero(np.argmax(red, axis=0))[0]
    y = np.nonzero(np.argmax(red, axis=1))[0]
    start_x = int(sum(x) / len(x))
    start_y = int(sum(y) / len(y))

    return start_x, start_y


def get_end_point_coordinates(img):
    blue = cv2.inRange(img, (204, 69, 57), (204, 155, 182))
    x = np.nonzero(np.argmax(blue, axis=0))[0]
    y = np.nonzero(np.argmax(blue, axis=1))[0]
    end_x = int(sum(x) / len(x))
    end_y = int(sum(y) / len(y))

    return end_x, end_y


def find_near_coordinate(x, y):
    need_point = 0
    rast = 2000
    for point, coord in dots.items():
        x2, y2 = coord
        ab = ((x2 - x) ** 2 + (y2 - y) ** 2) ** 0.5
        if ab < rast:
            rast = ab
            need_point = point
    return need_point


dots = {
    18: (1192, 824), 13: (352, 774), 19: (1184, 744), 12: (352, 690),
    3: (860, 623), 4: (627, 622), 2: (915, 572),
    5: (566, 570), 16: (1431, 488), 17: (1351, 485), 10: (81, 462),
    11: (161, 459), 1: (915, 339), 6: (560, 337),
    7: (620, 288), 0: (866, 279), 9: (348, 220), 15: (1161, 173), 8: (342, 134),
    14: (1155, 89)}

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

input_graph = {0: 5, 1: 15, 2: 7, 3: 18, 4: 1, 5: 12, 6: 3, 7: 8, 8: 10, 9: 6,
               10: 13, 11: 9, 12: 11, 13: 4, 14: 0,
               15: 17, 16: 14, 17: 19, 18: 16, 19: 2}
reverse_graph = {5: 0, 15: 1, 7: 2, 18: 3, 1: 4, 12: 5, 3: 6, 8: 7, 10: 8, 6: 9,
                 13: 10, 9: 11, 11: 12, 4: 13, 0: 14, 17: 15, 14: 16, 19: 17,
                 16: 18, 2: 19}
ind_per = ['05', '27', '41', '63']

pi, ESC, STEER = setup_gpio()
angle = 90

control(1500, 90)
time.sleep(1.5)

cap = cv2.VideoCapture(0)

perspective = np.array([])

flag = False
t0 = 0
count = 0

img = cv2.imread('1.png')

end_x, end_y = get_end_point_coordinates(img)
start_x, start_y = get_start_point_coordinates(img)
start = find_near_coordinate(start_x, start_y)
end = find_near_coordinate(end_x, end_y)
if start == end:
    x, y = dots[reverse_graph[start]]
    if ((start_x - x) ** 2 + (start_y - y) ** 2) ** 0.5 < (
            (end_x - x) ** 2 + (end_y - y) ** 2) ** 0.5:
        start = reverse_graph[start]
    else:
        end = reverse_graph[start]
print(start, end)
need_count, path = get_path(start, end)
print('Кол-во перекрёстков:', need_count)
time.sleep(1)

while True:
    status, frame = cap.read()

    if status:
        if flag:
            angle = 84
        else:
            get_angle()
        control(1558, angle)
        if time.monotonic() - t0 >= 1.7 and not flag and np.sum(
                perspective[130:200, :]) >= 5400000:
            # print('-------------detect stop')
            control(1558, 84)
            flag = True
            t0 = time.monotonic()
        if flag and time.monotonic() - t0 >= 1.7:
            flag = False
            count += 1
        if count == need_count:
            break
