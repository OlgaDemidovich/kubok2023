import pandas as pd
import cv2
import os

csv_file = "annotations.csv"
data = pd.read_csv(csv_file, sep=',')
data = data.sample(frac=1)
class_names = ["road_works", "parking", "no_entry", "pedestrian_crossing",
               "movement_prohibition", "artificial_roughness", "give_way",
               "stop"]
for i, row in enumerate(data.itertuples()):
    row_id, image_filename, real_label = row

    image = cv2.imread(image_filename)
    ind = class_names.index(real_label)
    DIR = f'dataset/{ind}/'
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    name = len([n for n in os.listdir(DIR)])
    image_filename = f"dataset/{ind}/{name}.jpg"
    cv2.imwrite(image_filename, image)
