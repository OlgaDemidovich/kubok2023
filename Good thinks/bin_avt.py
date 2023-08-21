import cv2


def nothing():
    pass


cv2.namedWindow('Trackbar2')

cv2.createTrackbar('minb', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('ming', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('minr', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('maxb', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('maxg', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('maxr', 'Trackbar2', 0, 255, nothing)

Background_img = cv2.imread('a.jpg')
cv2.imshow('Trackbar2', Background_img)
cap = cv2.VideoCapture(0)

while True:
    minb = cv2.getTrackbarPos('minb', 'Trackbar2')
    ming = cv2.getTrackbarPos('ming', 'Trackbar2')
    minr = cv2.getTrackbarPos('minr', 'Trackbar2')
    maxb = cv2.getTrackbarPos('maxb', 'Trackbar2')
    maxg = cv2.getTrackbarPos('maxg', 'Trackbar2')
    maxr = cv2.getTrackbarPos('maxr', 'Trackbar2')
    status, frame = cap.read()
    if not status:
        continue

    print(len(frame), len(frame[0]))
    # frame = cv2.imread('testing/all.jpg')
    # frame = cv2.resize(frame, (1000, 1000))
    cv2.imshow('frame', frame)
    frameCopy = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # cv2.imshow('hsv', hsv)
    # hsv = cv2.blur(hsv, (5, 5))

    mask = cv2.inRange(hsv, (minb, ming, minr), (maxb, maxg, maxr))


    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    cv2.imshow('mask', mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            cv2.drawContours(frame, contour, -1, (255, 0, 255), 3)
            cv2.imshow('contours', frame)

            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('Rect', frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
