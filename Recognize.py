import random

from scipy import ndimage
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

import cv2
import numpy as np
import os
import pandas as pd
import json

from matplotlib import pyplot as plt
from sklearn.svm import SVC
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
import main


def segment_and_recognize(plate_image, svm, svm2, scaler):
    """
    In this file, you will define your own segment_and_recognize function.
    To do:
        1. Segment the plates character by character
        2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
        3. Recognize the character by comparing the distances
    Inputs:(One)
        1. plate_imgs: cropped plate images by Localization.plate_detection function
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Outputs:(One)
        1. recognized_plates: recognized plate characters
        type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
    Hints:
        You may need to define other functions.
    """
    # 1
    # k1 parameter is percentage of max size of a letter/digit from the image/k2 for the smallest
    # ratioStandard - percentage of the image that is a letter/digit

    ratioStandard = 0.3
    ratioMax = 0.9
    heighConstant = 0.2

    # def processEachDL(image, epsilon, k1, k2, ratioStandard):
    # listOfChars = processEachDL(plate_image, 0.1, 0.2, 0.01, ratioStandard)

    listOfChars = processEachDL(plate_image, 0.1, 0.3, 0.01, ratioStandard, ratioMax, heighConstant)

    #
    # print(f'len of lisOfChars: {len(listOfChars)}')
    # 2
    # First will try KNN and then different algos
    # we iterate thorugh letters then digits and calculate an error for that
    # using switch we add a letter
    # return a string
    # for c in listOfChars:
    # svm = fineTuneTrainSet(main.Cat1Train, "dataset/groundTruth_platesFileNames.csv", 0.1, 0.2, 0.01, ratioStandard)

    # TODO: uncomment if doesnt work

    plateString = ''
    for i in listOfChars:
        charFound = predictModel(svm2, i, compareSize=(64, 64), hogUse=False)
        # if( charFound =="D" or charFound =="0" ):
        #     charFound = predictModel(svm, i,scaler=scaler)
        # plotImage(i, f'recognized as: {charFound}')
        plateString += str(charFound)
        # plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        # plt.title(f'{charFound}')
        # plt.axis('off')
        # plt.show()
    print(plateString)
    return plateString

    # recognized_plates = [None, None, None]
    # return recognized_plates


# TODO: yse image aumentation if still necessary after fine tuning

# def imageAugmentation(originalImage, imageNumber,directory,  numberOfAugumentations = 10):
#     #TODO: save the image to the directory as its file number + 20
#     image = originalImage.reshape((1, *originalImage.shape, 1))
#     pass


def hogPreprocessing(image, visualize=False):
    features, hog_img = hog(
        image,
        orientations=16,
        pixels_per_cell=(4, 4),
        cells_per_block=(3, 3),
        block_norm='L2-Hys',
        feature_vector=True,
        visualize=True
    )

    # if visualize:
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image, cmap='gray')
    #     plt.title("Original Image")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(hog_img, cmap='gray')
    #     plt.title("HOG Visualization")
    #     plt.axis('off')

    # plt.show()
    return features


def predictModel(myModel, imageToClassify, compareSize=(64, 64), hogUse=True, scaler=None):
    img = imageToClassify
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    target_size = max(img.shape)

    proi = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - img.shape[0]) // 2
    x_offset = (target_size - img.shape[1]) // 2

    proi[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    imgProposition = cv2.resize(proi, compareSize, interpolation=cv2.INTER_AREA)
    # imgProposition = cleaningChar(imgProposition, (3, 3), True)
    # imageToClassify = imgProposition.flatten() / 255.
    if (hogUse):
        imageToClassify = hogPreprocessing(imgProposition, True)
    else:
        imageToClassify = imgProposition.flatten() / 255.

    imageToClassify = imageToClassify.flatten().reshape(1, -1)

    if scaler is not None:
        imageToClassify = scaler.transform(imageToClassify)

    y_predict = myModel.predict(imageToClassify)

    # TODO: check if it works
    if hogUse:
        # Filter out '1' and 'L' and return the top other value
        y_predict = [pred for pred in y_predict if pred not in ['1', 'L']]
        if y_predict:
            return max(y_predict, key=y_predict.count)
        else:
            return None

    return y_predict[0]


def pathToPlate(csvPath):
    df = pd.read_csv(csvPath)
    plate = {}
    for _, row in df.iterrows():
        file_names = row['file name (plate)'].split(", ")
        for file_name in file_names:
            plate[file_name] = row['License plate'].replace("-", "")
    return plate


# TODO: do this if HOG doesnt fix the issue with 0's and O's
def fineTuneTrainSet(Category, csvFile, epsilon, k1, k2, ratioStandard, compareSize=(64, 64)):
    svm, x_train_old, y_train_old = svmModel()

    plateMapper = pathToPlate(csvFile)

    x_train_new, y_train_new = [], []

    for filePath in Category:
        accuracy, image, lastResult = main.processJsonGetAccuracy(filePath)
        x, y, w, h = lastResult.x, lastResult.y, lastResult.w, lastResult.h
        listaOfChars = processEachDL(image[y:y + h, x:x + w], epsilon, k1, k2, ratioStandard)

        plate = plateMapper.get(filePath)

        for charImage, charLabel in zip(listaOfChars, plate):
            # resizedImage = cv2.resize(charImage, compareSize).flatten() / 255.0
            resizedImage = cv2.resize(charImage, compareSize, interpolation=cv2.INTER_AREA)
            # resizedImage = cleaningChar(resizedImage)
            # resizedImage = resizedImage.flatten() / 255.0
            resizedImage = hogPreprocessing(resizedImage)

            x_train_new.append(resizedImage)
            y_train_new.append(charLabel)

        x_train = x_train_old + x_train_new
        y_train = y_train_old + y_train_new

        svm.fit(x_train, y_train)

        return svm


def svmModel(compareSize=(64, 64), n_neighbors=7, knnDistance=False, useHog=True):
    # we have to generate a dataset
    x_train, y_train = [], []
    directoryToTravel = {}

    augmentedDir = "dataset/AugmentedImages"
    labelsFPath = os.path.join(augmentedDir, "labels")

    for i in range(1, (28)):
        xCurrent, yCurrent = getPhotoPath(i)
        directoryToTravel[i] = (xCurrent, yCurrent)

    if os.path.exists(labelsFPath):
        with open(labelsFPath, 'r') as labelF:
            for line in labelF:
                parts = line.strip().split()
                if len(parts) == 2:
                    imageName, label = parts
                    imagePath = os.path.join(augmentedDir, imageName)
                    directoryToTravel[len(directoryToTravel) + 1] = (imagePath, label)


    shuffled_items = list(directoryToTravel.items())
    random.shuffle(shuffled_items)

    for key, (xCurrent, yCurrent) in shuffled_items:

        img = cv2.imread(xCurrent)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #  detect a contour so there is no background stupid match
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y + h, x:x + w]

        target_size = max(roi.shape)

        proi = np.zeros((target_size, target_size), dtype=np.uint8)
        y_offset = (target_size - roi.shape[0]) // 2
        x_offset = (target_size - roi.shape[1]) // 2

        proi[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi

        imgProposition = cv2.resize(proi, compareSize, interpolation=cv2.INTER_NEAREST)
        # cleaning not necessary as these are cleaned
        # imgProposition = cleaningChar(imgProposition)
        # not necessary anymore as hog method is doing all that
        # imgProposition = imgProposition.flatten() / 255.

        # making sure that there is an outer black pixel frame
        imgProposition[0, :] = 0  # Top
        imgProposition[-1, :] = 0  # Bottom
        imgProposition[:, 0] = 0  # Left
        imgProposition[:, -1] = 0  # Right
        # TODO:preprocess hog here xCurrent
        if (useHog):
            imgProposition = hogPreprocessing(imgProposition, False)
        else:
            imgProposition = imgProposition.flatten() / 255.

        x_train.append(imgProposition)
        y_train.append(yCurrent)

    if (useHog):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

    # we have to train SVM model

    # svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')

    # svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    if (knnDistance):
        knn = KNeighborsClassifier(n_neighbors, weights='distance')
    else:
        knn = KNeighborsClassifier(n_neighbors)
    knn.fit(x_train, y_train)

    if (useHog):
        return knn, x_train, y_train, scaler
    return knn, x_train, y_train


def plotImage(img, title, cmapType=None):
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def bestDistance(charPhoto, compareSize=(64, 64)):
    if len(charPhoto.shape) == 3:
        charPhoto = cv2.cvtColor(charPhoto, cv2.COLOR_BGR2GRAY)

    charPhotoResized = cv2.resize(charPhoto, compareSize)
    charTruth = charPhotoResized.flatten() / 255.

    topDistance = float('inf')
    topLetterDigit = None

    for i in range(1, 28):

        fullPath, label = getPhotoPath(i);

        img = cv2.imread(fullPath)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #  detect a contour so there is no background stupid match
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y + h, x:x + w]

        imgProposition = cv2.resize(roi, compareSize)
        imgProposition = imgProposition.flatten() / 255.

        value = euclideanDistance(imgProposition, charTruth)

        if (value < topDistance):
            topLetterDigit = label
            topDistance = value

    return topLetterDigit, topDistance


def getPhotoPath(i):
    pathLetters = "dataset/SameSizeLetters"
    pathDigits = "dataset/SameSizeNumbers"
    fullPath = None
    label = None
    if (i < 18):
        fullPath = os.path.join(pathLetters, f'{i}.bmp')
        label = assignLetter(i)

    if (i >= 18):
        temp = i - 18
        fullPath = os.path.join(pathDigits, f'{temp}.bmp')
        label = str(temp)
    return fullPath, label


def assignLetter(value):
    cases = {
        1: "B",
        2: "D",
        3: "F",
        4: "G",
        5: "H",
        6: "J",
        7: "K",
        8: "L",
        9: "M",
        10: "N",
        11: "P",
        12: "R",
        13: "S",
        14: "T",
        15: "V",
        16: "X",
        17: "Z"
    }
    return cases.get(value)


def euclideanDistance(img1, img2):
    distance = np.sqrt(np.sum((img1 - img2) ** 2))

    return distance


# from the weekly assignments
def contrastImprovementContrastStretching(img, a=None, b=None, title=None):
    # Do point operation on pixels
    img = img.astype(np.float32)

    if a is None or b is None:
        IMin, IMax = np.min(img), np.max(img)
        a = 255.0 / (IMax - IMin)
        b = -IMin

    img = a * (img + b)

    #
    img = img.astype(np.uint8)

    # Return the result
    return img


def adjustContrast(img, alpha=1.2, beta=0):

    img = img.astype(np.float32)
    img = 255 / (1 + np.exp(-alpha * (img / 255.0 - 0.5)))
    img = img + beta
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img





def findEdges(mask, minSize=(450, 190), stackCount=2, debug=False, debugLast = False):
    orig_h, orig_w = mask.shape[:2]
    if orig_w < minSize[0] or orig_h < minSize[1]:
        scaleX = minSize[0] / orig_w
        scaleY = minSize[1] / orig_h
        scale = max(scaleX, scaleY)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if debugLast:
        plt.imshow(mask, cmap='gray')
        plt.title('Mask start')
        plt.axis('off')
        plt.show()

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    if debug:
        plt.imshow(mask_closed, cmap='gray')
        plt.title('Closing')
        plt.axis('off')
        plt.show()

    # TODO: redistribution
    redistributed = contrastImprovementContrastStretching(mask_closed);

    if debugLast:
        plt.imshow(redistributed, cmap='gray')
        plt.title('Redistributed')
        plt.axis('off')
        plt.show()


    stackCount =2
    stacked = np.zeros_like(redistributed, dtype=np.float32)
    for _ in range(stackCount):
        stacked += redistributed.astype(np.float32)

    # stacked /= stackCount

    stacked = np.clip(stacked, 0, 255).astype(np.uint8)

    if debugLast:
        plt.imshow(stacked, cmap='gray')
        plt.title(" Stacked")
        plt.axis('off')
        plt.show()


    # #TODO: make closing
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # stacked = cv2.morphologyEx(stacked, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    #
    # if debug:
    #     plt.imshow(mask_closed, cmap='gray')
    #     plt.title('Second Closing')
    #     plt.axis('off')
    #     plt.show()
    # #finished

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    stacked_clahe = clahe.apply(stacked)

    if debug:
        plt.imshow(stacked_clahe, cmap='gray')
        plt.title("CLAHE")
        plt.axis('off')
        plt.show()

    blurred = cv2.GaussianBlur(stacked_clahe, (3, 3), 0)
    sharpened = cv2.addWeighted(stacked_clahe, 1.5, blurred, -0.5, 0)

    if debug:
        plt.imshow(sharpened, cmap='gray')
        plt.title('AFTER SHARPENING ')
        plt.axis('off')
        plt.show()

    thresh = isodata_thresholding(sharpened)

    thresh_inv = 255 - thresh

    if debug:
        plt.imshow(thresh_inv, cmap='gray')
        plt.title(' DONe treshold isodata_thresholding ')
        plt.axis('off')
        plt.show()

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel_open, iterations=2)

    if debug:
        plt.imshow(opened, cmap='gray')
        plt.title(' OPENIN G')
        plt.axis('off')
        plt.show()

    kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close2, iterations=2)

    if debug:
        plt.imshow(closed, cmap='gray')
        plt.title('CLOSING ')
        plt.axis('off')
        plt.show()

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    if debug:
        plt.imshow(dilated, cmap='gray')
        plt.title(' thinckening layers ')
        plt.axis('off')
        plt.show()

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)

    if debug:
        plt.imshow(eroded, cmap='gray')
        plt.title('Eroded')
        plt.axis('off')
        plt.show()

    final = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close2, iterations=2)

    if debugLast:
        plt.imshow(final, cmap='gray')
        plt.title('Final Result!!!! <3')
        plt.axis('off')
        plt.show()

    return final


# returns an array of Digits/Letters
def processEachDL(image, epsilon, k1, k2, ratioStandard, ratioMax, heighConstant):
    listaChars = []

    # We convert RGB->Gray
    # imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if len(image.shape) > 2 and image.shape[2] > 1:
        imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        imgGray = image.copy()

    # separatin into a binary image
    image = isodata_thresholding(imgGray, epsilon)
    # inverting so letter/digits are white the plate is black
    imageInverted = 255 - image

    # Calculate an area of the image so then we can use that to discard not valid contours
    height, width = image.shape[:2]
    MAX_AREA = height * width



    # Now if we have find countours for each digit
    contours, _ = cv2.findContours(imageInverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flag = False
    # if (len(contours) <6):
    if (True):
        flag = True
        # check if there are contours??? if no then go to

        imageInverted = findEdges(imgGray)
        contours, _ = cv2.findContours(imageInverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = imageInverted.shape[:2]
        MAX_AREA = height * width
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        areaHere = w * h

        if (flag):
            # LAST
            # plt.imshow(cv2.cvtColor(imageInverted[y:y + h, x:x + w], cv2.COLOR_GRAY2RGB))
            # plt.title('ROI IMAGE INVERTED INSIDE contour extraction ')
            # plt.axis('off')
            # plt.show()
            pass

        # if this contour is too big we discard it

        if (True):

            if not (MAX_AREA * k2 < areaHere < MAX_AREA * k1):
                continue

            # ratio

            roi = imageInverted[y:y + h, x:x + w]

            nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
            allPixelsInTheRectangle = w * h
            ratio = nonZeroPixelsInTheRectangle / allPixelsInTheRectangle

            if (ratioMax < ratio < ratioStandard):
                continue

            if h < heighConstant * height:
                continue

            heightWidthRatio = h / w
            if not (1. <= heightWidthRatio <= 5.):
                continue

            imageCropped = imageInverted[y:y + h, x:x + w]
            listaChars.append(imageCropped)

    return listaChars


def isodata_thresholding(image, epsilon=2):
    # Compute the histogram and set up variables
    hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2 * epsilon
    # Iterations of the isodata thresholding algorithm
    while (abs(tau - old_tau) >= epsilon):
        ForegroundMask = image >= tau

        m1 = image[ForegroundMask]
        # this so there is no division by zero problem
        m1Len = np.count_nonzero(m1) if np.count_nonzero(m1) != 0 else 0.0001
        m1Sum = np.sum(m1)
        m1 = m1Sum / m1Len

        BackgroundMask = image < tau
        m2 = image[BackgroundMask]
        m2Len = np.count_nonzero(m2) if np.count_nonzero(m2) != 0 else 0.0001
        m2Sum = np.sum(m2)
        m2 = m2Sum / m2Len

        old_tau = tau
        tau = (m1 + m2) / 2

    # background = np.where(image < tau, image, 0)
    # foreground = np.where(image >= tau, image, 0)
    image = np.where(image >= tau, 255, 0).astype(np.uint8)

    # tau = max(0, min(255, tau))
    #
    #
    # binary_image = cv2.threshold(image, tau, 255, cv2.THRESH_BINARY)[1]
    return image


if __name__ == "__main__":
    plate_image_path = "plate4.png"
    plate_image = cv2.imread(plate_image_path)
    # svm2, _, _ = svmModel(compareSize=(64, 64), n_neighbors=2, knnDistance=True, useHog=False)
    #
    # svm, x_train, y_train, scaler = svmModel()

    # joblib.dump(svm, 'svm_model.pkl')
    # joblib.dump(svm2, 'svm2_model.pkl')
    # joblib.dump(scaler, 'scaler.pkl')

    svm, svm2, scaler = joblib.load('svm_model.pkl'), joblib.load('svm2_model.pkl'), joblib.load('scaler.pkl')

    if plate_image is None:
        print(f"Error: Could not load image from path {plate_image_path}")
    else:
        segment_and_recognize(plate_image, svm, svm2, scaler)
