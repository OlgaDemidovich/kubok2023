# -*- coding: utf-8 -*-
"""
Файл служит для определения точности вашего алгоритма

Для получения оценки точности, запустите файл на исполнение
"""

import eval as submission
import cv2
import pandas as pd


def main():
    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=',')
    data = data.sample(frac=1)

    models = submission.load_models()

    correct = 0
    for i, row in enumerate(data.itertuples()):
        row_id, image_filename, real_label = row

        image = cv2.imread(image_filename)

        user_answer = submission.predict_sign(image, models)

        if real_label == user_answer:
            correct += 1
            print(image_filename, '- верно')
        else:
            print(image_filename, '- неверно')

    total_object = len(data.index)
    print(f"Из {total_object} знаков верно классифицированы {correct}")

    score = correct / total_object
    print(f"Точность: {score:.2f}")


if __name__ == '__main__':
    main()
