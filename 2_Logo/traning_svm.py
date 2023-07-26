import pandas as pd
import os
import cv2
import dlib

dir = r'D:/inbox/Документы/Документы/__Python__/kubok2023/2_Logo/'
# dir = r'kubok2023/2_Logo'


csv_file = "annotations.csv"
data = pd.read_csv(csv_file, sep=',')
data = data.sample(frac=1)
names = {'cpp': 0, 'python': 1, 'kruzhok': 2, 'altair': 3, 'avt': 4}

correct = 0
for name in ['cpp', 'python', 'kruzhok', 'altair', 'avt']:
    images = []
    annots = []
    for i, row in enumerate(data.itertuples()):
        row_id, image_filename, real_label, x, y, w, h = row
        x, y, x2, y2 = x, y, w+x, h+y

        if real_label != name:
            continue
        image = cv2.imread(image_filename)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_viz = image.copy()

        # cv2.rectangle(image, (x, y), (x2, y2), (0, 200, 0))  # коммент
        # cv2.imshow('image', image)  # коммент
        # cv2.waitKey(0)  # коммент

        images.append(image)
        annots.append(
            [dlib.rectangle(left=x, top=y, right=x2, bottom=y2)])

    options = dlib.simple_object_detector_training_options()
    options.be_verbose = True

    detector = dlib.train_simple_object_detector(images, annots, options)

    detector.save(f'Detector_logo_{name}.svm')
print('All')
