import random

from scipy import ndimage
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


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


def segment_and_recognize(plate_image,svm,svm2,scaler):
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
    #TODO: uncomment
    ratioStandard = 0.3
    ratioMax = 0.9
    heighConstant = 0.2

    # def processEachDL(image, epsilon, k1, k2, ratioStandard):
    # listOfChars = processEachDL(plate_image, 0.1, 0.2, 0.01, ratioStandard)

    listOfChars = processEachDL(plate_image, 0.1, 0.3, 0.01, ratioStandard,ratioMax,heighConstant)

    #
    # print(f'len of lisOfChars: {len(listOfChars)}')
    # 2
    # First will try KNN and then different algos
    # we iterate thorugh letters then digits and calculate an error for that
    # using switch we add a letter
    # return a string
    # for c in listOfChars:
    # svm = fineTuneTrainSet(main.Cat1Train, "dataset/groundTruth_platesFileNames.csv", 0.1, 0.2, 0.01, ratioStandard)

    #TODO: uncomment if doesnt work

    plateString = ''
    for i in listOfChars:

        charFound = predictModel(svm2, i, compareSize=(64,64), hogUse =False)
        if( charFound =="D" or charFound =="0" ):
            charFound = predictModel(svm, i,scaler=scaler)
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


def rotateImage(image):
    angle = calculateRotationAngle(image)
    image_array = np.array(image)
    rotated = ndimage.rotate(image_array, angle)
    plt.imshow(rotated)

    plt.axis('off')

    plt.show()
    return rotated

def calculateRotationAngle(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Ensure grayscale image is of type uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    # Apply Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest rectangle-like contour (License plate candidate)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour) # Get min area bounding box
    # Get rotation angle
    angle = rect[-1]
    # Adjust the angle to ensure correct orientation
    if angle < -45:
        angle += 90

    return angle -90





def hogPreprocessing(image, visualize =False):
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


def predictModel(myModel, imageToClassify, compareSize=(64,64), hogUse =True, scaler=None):


    img = imageToClassify
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    target_size = max(img.shape)

    proi = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - img.shape[0]) // 2
    x_offset = (target_size - img.shape[1]) // 2

    proi[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    imgProposition = cv2.resize(proi, compareSize,interpolation=cv2.INTER_AREA)
    # imgProposition = cleaningChar(imgProposition, (3, 3), True)
    # imageToClassify = imgProposition.flatten() / 255.
    if (hogUse):
        imageToClassify = hogPreprocessing(imgProposition,True)
    else:
        imageToClassify = imgProposition.flatten() / 255.


    imageToClassify = imageToClassify.flatten().reshape(1, -1)

    if scaler is not None:
        imageToClassify = scaler.transform(imageToClassify)

    y_predict = myModel.predict(imageToClassify)


    #TODO: check if it works
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
def fineTuneTrainSet(Category, csvFile, epsilon, k1, k2, ratioStandard, compareSize=(64,64)):
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
            resizedImage = cv2.resize(charImage, compareSize,interpolation=cv2.INTER_AREA)
            # resizedImage = cleaningChar(resizedImage)
            # resizedImage = resizedImage.flatten() / 255.0
            resizedImage = hogPreprocessing(resizedImage)

            x_train_new.append(resizedImage)
            y_train_new.append(charLabel)

        x_train = x_train_old + x_train_new
        y_train = y_train_old + y_train_new

        svm.fit(x_train, y_train)

        return svm


def svmModel(compareSize=(64,64), n_neighbors=7,knnDistance =False, useHog = True):
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

    print(directoryToTravel)

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

        imgProposition = cv2.resize(proi, compareSize,interpolation=cv2.INTER_NEAREST)
        # cleaning not necessary as these are cleaned
        # imgProposition = cleaningChar(imgProposition)
        # not necessary anymore as hog method is doing all that
        # imgProposition = imgProposition.flatten() / 255.

        #making sure that there is an outer black pixel frame
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

    if(useHog):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)



    # we have to train SVM model

    # svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')

    # svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    if(knnDistance):
        knn = KNeighborsClassifier(n_neighbors, weights='distance')
    else:
        knn = KNeighborsClassifier(n_neighbors)
    knn.fit(x_train, y_train)

    if(useHog):
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

def findEdges(mask, minSize=(300, 100), stackCount=3):

    if mask.shape[1] < minSize[0] or mask.shape[0] < minSize[1]:
        scaleX = minSize[0] / mask.shape[1]
        scaleY = minSize[1] / mask.shape[0]
        scale = max(scaleX, scaleY)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)



    stacked = np.zeros_like(mask, dtype=np.float32)
    for _ in range(stackCount):
        stacked += mask.astype(np.float32)
    stacked = np.clip(stacked, 0, 255).astype(np.uint8)

    stacked= 10*stacked



    plt.imshow(cv2.cvtColor(stacked, cv2.COLOR_GRAY2RGB))
    plt.title('Stacked')
    plt.axis('off')
    plt.show()
    _, thresh = cv2.threshold(stacked, 120, 255, cv2.THRESH_BINARY)


    thresh = 255 - thresh


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    plt.imshow(cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB))
    plt.title('Closed image - fill in the gaps')
    plt.axis('off')
    plt.show()


    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    plt.imshow(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB))
    plt.title('Cleaned Black Letters on White BG')
    plt.axis('off')
    plt.show()

    return cleaned


# returns an array of Digits/Letters
def processEachDL(image, epsilon, k1, k2, ratioStandard,ratioMax, heighConstant):
    # 1
    # image = rotateImage(image)
    listaChars = []

    # We convert RGB->Gray
    imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(cv2.cvtColor(imgGray,cv2.COLOR_GRAY2RGB))
    plt.title('after rgb2gray')
    plt.axis('off')
    plt.show()



    # separatin into a binary image
    #TODO: uncomment
    image = isodata_thresholding(imgGray, epsilon)
    # inverting so letter/digits are white the plate is black
    imageInverted = imgGray -255
    plt.imshow(cv2.cvtColor(imageInverted,cv2.COLOR_GRAY2RGB))
    plt.title('after is o ')
    plt.axis('off')
    plt.show()

    # Calculate an area of the image so then we can use that to discard not valid contours
    height, width = image.shape[:2]
    MAX_AREA = height * width
    # print(f'Max Area: {MAX_AREA}')
    # #
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title(f'{MAX_AREA}')
    # plt.axis('off')
    # plt.show()





    # Now if we have find countours for each digit
    contours, _ = cv2.findContours(imageInverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 0):
        # check if there are contours??? if no then go to
        imageInverted = findEdges(imgGray)
        plt.imshow(cv2.cvtColor(imageInverted, cv2.COLOR_GRAY2RGB))
        plt.title('AFTER Edges doesnt count lol ')
        plt.axis('off')
        plt.show()
    for contour in contours:
        area = cv2.contourArea(contour)


        x, y, w, h = cv2.boundingRect(contour)

        areaHere = w*h

        print(f'contour Are: {areaHere}')
        # if this contour is too big we discard it
        if not (MAX_AREA * k2 < areaHere < MAX_AREA * k1):
            continue

        # ratio
        roi = imageInverted[y:y + h, x:x + w]

        nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
        allPixelsInTheRectangle = w * h
        ratio = nonZeroPixelsInTheRectangle / allPixelsInTheRectangle



        if (ratioMax<ratio < ratioStandard):
            continue


        if h < heighConstant * height:
            continue

        imageCropped = imageInverted[y:y + h, x:x + w]
        listaChars.append(imageCropped)

        # visualization
        # plt.imshow(cv2.cvtColor(imageCropped, cv2.COLOR_GRAY2RGB))
        # plt.title(f'{area}')
        # plt.axis('off')
        # plt.show()

    #
    # plt.title('Green')
    # plt.axis('off')
    # plt.show()
    return listaChars


def isodata_thresholding(image, epsilon=2):
    # Compute the histogram and set up variables
    hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2 * epsilon
    # Iterations of the isodata thresholding algorithm
    while (abs(tau - old_tau) >= epsilon):
        ForegroundMask = image >= tau
        # TODO Calculate m1
        m1 = image[ForegroundMask]
        # this so there is no division by zero problem
        m1Len = np.count_nonzero(m1) if np.count_nonzero(m1) != 0 else 0.0001
        m1Sum = np.sum(m1)
        m1 = m1Sum / m1Len
        # TODO Calculate m2
        BackgroundMask = image < tau
        m2 = image[BackgroundMask]
        m2Len = np.count_nonzero(m2) if np.count_nonzero(m2) != 0 else 0.0001
        m2Sum = np.sum(m2)
        m2 = m2Sum / m2Len
        # TODO Calculate new tau
        old_tau = tau
        tau = (m1 + m2) / 2
    # TODO Threshold the image based on last tau
    # background = np.where(image < tau, image, 0)
    # foreground = np.where(image >= tau, image, 0)
    image = np.where(image >= tau, 255, 0).astype(np.uint8)

    # tau = max(0, min(255, tau))
    #
    #
    # binary_image = cv2.threshold(image, tau, 255, cv2.THRESH_BINARY)[1]
    return image





if __name__ == "__main__":
    plate_image_path = "plate3.png"
    plate_image = cv2.imread(plate_image_path)
    svm2, _, _ = svmModel(compareSize=(64, 64), n_neighbors=2, knnDistance=True, useHog=False)

    svm, x_train, y_train, scaler = svmModel()
    if plate_image is None:
        print(f"Error: Could not load image from path {plate_image_path}")
    else:
        segment_and_recognize(plate_image,svm,svm2,scaler)


