import cv2
import numpy as np
from matplotlib import pyplot as plt


def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    # plate = yellowMask(image)
    # plate = whiteMask(image)
    plate = blackMask(image)
    return plate
    # TODO: Replace the below lines with your code.
    # plate_images = [image, image, image]
    # return plate_images
    pass


def yellowMask(image):
    lower = np.array([12, 70, 60])
    upper = np.array([35,255,255])

    return applyMask(image, lower, upper)



def blackMask(image):
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])

    return applyMask(image, lower, upper)

def whiteMask(image):
   lower = np.array([0, 0, 200])           
   upper = np.array([180, 55, 255])

   return applyMask(image, lower, upper)

def applyMask(image, lower, upper):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image, lower, upper)

    image = cv2.bitwise_and(image,image, mask=mask)

    return cropPlate(image, mask)

def cropPlate(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rectangle = cv2.contourArea(contour)
        if rectangle < 700:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        counter = 30;

        flag = True
        while flag and counter>0:
            roi = mask[y:y + h, x:x + w]

            #already cropped plates but I leave it here since it might be useful later
            # nonZeroRectangle = cv2.findNonZero(roi)
            # nx, ny, nw, nh = cv2.boundingRect(nonZeroRectangle)
            # x, y, w, h = x + nx, y + ny, nw, nh

            nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
            allPixelsInTheRectangle = w * h
            ratio = 0 if allPixelsInTheRectangle == 0 else nonZeroPixelsInTheRectangle / allPixelsInTheRectangle



            if ratio >= 0.5 :
                flag = False  # Valid ratio, below .5 weirdly situated plates are not included

            else:
                counter=counter-1;
                x += 1
                y += 1
                w -= 2
                h -= 2


    plateAfterCrop = image[y:y + h, x:x + w]

    plateAfterCrop = cv2.cvtColor(plateAfterCrop, cv2.COLOR_BGR2RGB)

    # cv2.imshow("Image After Crop", plateAfterCrop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return plateAfterCrop
