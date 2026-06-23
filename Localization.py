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
    # plate = blackMask(image)

    # return plate
    # TODO: Replace the below lines with your code.
    # plate_images = [image, image, image]
    # return plate_imag#



    # cv2.imshow("Image After Crop", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return yellowMask(image)


# work in progress
    # plates = detectMaskAnyColour(image)
    #
    # if plates:
    #     for m in plates:
    #         if isinstance(m, np.ndarray) and m.size > 0:  # Validate that `m` is a valid numpy array
    #             cv2.imshow("Detected Plate", m)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #         else:
    #             print("Skipped invalid plate data.")
    # else:
    #     print("No plates detected.")
    # pass


def yellowMask(image):
    lower = np.array([12, 70, 60])
    upper = np.array([35,255,255])

    return applyMask(image, lower, upper)



def blackMask(image):
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 65])

    return applyMask(image, lower, upper)

def whiteMask(image):
   lower = np.array([0, 0, 200])
   upper = np.array([180, 55, 255])

   return applyMask(image, lower, upper)

def applyMask(image, lower, upper):
    image_hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_hue, lower, upper)

    return cropPlate(image, mask,0.6,1.1,.4)


def cropPlate(image, mask, k1, k2, ratioFix):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plateAfterCrop = None
    x, y, w, h = 0, 0, 0, 0

    MIN_AREA = k1 * 2400
    MAX_AREA = k2 * 90000

    listaOfContours = []


    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:

            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi = mask[y:y + h, x:x + w]

        nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
        allPixelsInTheRectangle = w * h

        ratio = 0 if allPixelsInTheRectangle == 0 else nonZeroPixelsInTheRectangle / allPixelsInTheRectangle

        if ratio <= ratioFix:
            break
        plateAfterCrop = image[y:y + h, x:x + w]

        newContour = contourObject(plateAfterCrop,x,y,w,h)
        listaOfContours.append(newContour)

    return listaOfContours




# def detectMaskAnyColour(image):
#     plates = []
#     for mask in [yellowMask, whiteMask, blackMask]:
#         mask_results, _ = mask(image)
#         if mask_results:
#             for plate in mask_results:
#                 if isinstance(plate, np.ndarray) and plate.size > 0:
#                     plates.append(plate)
#
#     return plates


class contourObject():
    def __init__(self, croppedImage, x, y,w,h):
        self.croppedImage = croppedImage
        self.x =x
        self.y =y
        self.w = w
        self.h = h


    def getCroppedImage(self):
        return self.croppedImage

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

