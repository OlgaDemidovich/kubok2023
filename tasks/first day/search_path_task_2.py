import cv2
import numpy as np

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


input_graph = {0: 5, 1: 15, 2: 7, 3: 18, 4: 1, 5: 12, 6: 3, 7: 8, 8: 10, 9: 6,
               10: 13, 11: 9, 12: 11, 13: 4, 14: 0,
               15: 17, 16: 14, 17: 19, 18: 16, 19: 2}
reverse_graph = {5: 0, 15: 1, 7: 2, 18: 3, 1: 4, 12: 5, 3: 6, 8: 7, 10: 8, 6: 9,
                 13: 10, 9: 11, 11: 12, 4: 13, 0: 14, 17: 15, 14: 16, 19: 17,
                 16: 18, 2: 19}
ind_per = ['05', '27', '41', '63']


img = cv2.imread('Pub_NSK_Cup23_OflineChapter/images_2_task/2.png')

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

