import cv2

for i in range(1, 7):
    name = f'{i}.png'
    frame = cv2.imread('reference/'+name)
    # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)

    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 8)

    thresh = cv2.inRange(blurred_frame, (0, 0, 0), (100, 100, 100))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
        # cv2.rectangle(frame, bbox, (255, 0, 0), 1)
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        nut = thresh[y:y2, x:x2]
    cv2.imwrite(f'reference/_{name}', nut)
