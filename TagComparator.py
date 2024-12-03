import cv2
import numpy as np


def isClose(firstPoint, secondPoint):
    if (abs(firstPoint.getX() - secondPoint.getX()) > 20
            or abs(firstPoint.getY() - secondPoint.getY()) > 20):
        return False
    return True


def compareBoxes(box1, box2):
    for i in range(4):
        if (isClose(box1[i], box2[i])):
            return False

    return True


def captureTestVideo():
    path = "dataset/trainingvideo.avi"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps * 2)

    count = 1
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Process only the frames at the specified interval
        if count % frame_interval == 0:
            yellowMask = yellowMask(frame)
            x, y, w, h = boxCoordinates(yellowMask)

        count += 1


def yellowMask(image):
    lower = np.array([12, 70, 60])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(image, lower, upper)

    return mask


def boxCoordinates(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Image After Crop", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    MIN_AREA = 1500
    for contour in contours:

        rectangle = cv2.contourArea(contour)
        if rectangle < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        counter = 30;

        flag = True
        while flag and counter > 0:
            roi = mask[y:y + h, x:x + w]
            # Testing purposes
            # cv2.imshow("Image After Crop", roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # already cropped plates but I leave it here since it might be useful later
            # nonZeroRectangle = cv2.findNonZero(roi)
            # nx, ny, nw, nh = cv2.boundingRect(nonZeroRectangle)
            # x, y, w, h = x + nx, y + ny, nw, nh

            nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
            allPixelsInTheRectangle = w * h
            ratio = 0 if allPixelsInTheRectangle == 0 else nonZeroPixelsInTheRectangle / allPixelsInTheRectangle

            if ratio >= 0.5:
                flag = False  # Valid ratio, below .5 weirdly situated plates are not included

            else:
                counter = counter - 1;
                x += 1
                y += 1
                w -= 2
                h -= 2
    return x, y, w, h


captureTestVideo()
