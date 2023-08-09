import cv2
import pandas as pd
import os


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    return str(class_index) + " " + str(x_center) \
           + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)


def save_bb(txt_path, line):
    with open(txt_path, 'a') as myfile:
        myfile.write(line + "\n")  # append line


dir = r'D:/inbox/Документы/Документы/__Python__/kubok2023/4_Traffic_light/'
csv_file = "annotations.csv"
data = pd.read_csv(dir+csv_file, sep=',')
data = data.sample(frac=1)

# img_dir = "images/"
# image_list = []
# for f in os.listdir(img_dir):
#     f_path = os.path.join(img_dir, f)
#     test_img = cv2.imread(f_path)
#     if test_img is not None:
#         image_list.append(f_path)
#
# for img_path in image_list:
#     txt_path = get_txt_path(img_path)
#     if not os.path.isfile(txt_path):
#         open(txt_path, 'a').close()


correct = 0
images = []
annots = []
class_index = 0


for i, row in enumerate(data.itertuples()):
    row_id, image_filename, class_name, x, y, w, h = row
    image = cv2.imread(image_filename)
    height, width, _ = image.shape
    x, y, x2, y2 = x, y, (w + x), (h + y)
    point_1 = [x, y]
    point_2 = [x2, y2]
    line = yolo_format(class_index, point_1, point_2, width, height)
    # print(dir + image_filename.replace('.jpg', '.txt'))
    txt_path = dir + image_filename.replace('.jpg', '.txt').replace('images',
                                                                    'labels')
    if not os.path.isfile(txt_path):
        open(txt_path, 'a').close()
    save_bb(txt_path, line)
