from skimage.feature import hog
from skimage.io import imread

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


def segment_and_recognize(plate_images):
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
    #1
    #k1 parameter is percentage of max size of a letter/digit from the image/k2 for the smallest
    #ratioStandard - percentage of the image that is a letter/digit
    ratioStandard = 0.3
    listOfChars = processEachDL(plate_images,0.1,0.2,0.01,ratioStandard)

    #2
    #First will try KNN and then different algos
    #we iterate thorugh letters then digits and calculate an error for that
    # using switch we add a letter
    #return a string
    # for c in listOfChars:
    print(main.Cat1Train)
    svm= fineTuneTrainSet(main.Cat1Train,"dataset/groundTruth_platesFileNames.csv",0.1,0.2,0.01,ratioStandard )

    plateString = ''
    for i in listOfChars:

        # charFound, distance = bestDistance(i)
        charFound = predictModel(svm,i)
        plotImage(i,f'recognized as: {charFound}')
        plateString+=str(charFound)
    print(plateString)


    # recognized_plates = [None, None, None]
    # return recognized_plates

#TODO: yse image aumentation if still necessary after fine tuning

# def imageAugmentation(originalImage, imageNumber,directory,  numberOfAugumentations = 10):
#     #TODO: save the image to the directory as its file number + 20
#     image = originalImage.reshape((1, *originalImage.shape, 1))
#     pass

#Morphological operations
def cleaningChar():



def hogPreprocessing(image):
    features = hog(
        image,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features



def predictModel(myModel,imageToClassify,compareSize=(64,64)):
    img = imageToClassify
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgProposition = cv2.resize(img, compareSize)
    imageToClassify = imgProposition.flatten() / 255.
    # imageToClassify = hogPreprocessing(imgProposition)

    imageToClassify = imageToClassify.reshape(1, -1)

    y_predict = myModel.predict(imageToClassify)
    return y_predict[0]



def pathToPlate(csvPath):
    df = pd.read_csv(csvPath)
    plate = {}
    for _, row in df.iterrows():
        file_names = row['file name (plate)'].split(", ")
        for file_name in file_names:
            plate[file_name] = row['License plate'].replace("-", "")
    return plate


#TODO: do this if HOG doesnt fix the issue with 0's and O's
def fineTuneTrainSet( Category,csvFile, epsilon,k1,k2,ratioStandard, compareSize=(64,64)):

    svm,x_train_old,y_train_old = svmModel()

    plateMapper = pathToPlate(csvFile)

    x_train_new, y_train_new =[],[]

    for filePath in Category:
        accuracy, image,lastResult = main.processJsonGetAccuracy(filePath)
        x, y, w, h = lastResult.x, lastResult.y, lastResult.w, lastResult.h
        listaOfChars = processEachDL(image[y:y + h, x:x + w],epsilon,k1,k2,ratioStandard)

        plate = plateMapper.get(filePath)


        for charImage, charLabel in zip(listaOfChars, plate):

            resizedImage = cv2.resize(charImage, compareSize).flatten() / 255.0
            # resizedImage = cv2.resize(charImage,compareSize)
            # resizedImage = hogPreprocessing(resizedImage)

            x_train_new.append(resizedImage)
            y_train_new.append(charLabel)


        x_train =x_train_old+x_train_new
        y_train = y_train_old +y_train_new

        svm.fit(x_train,y_train)

        return svm


def svmModel(compareSize=(64,64)):
    #we have to generate a dataset
    x_train,y_train = [],[]
    for i in range(1, 28):


        xCurrent,yCurrent  = getPhotoPath(i)

        img = cv2.imread(xCurrent)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #  detect a contour so there is no background stupid match
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y + h, x:x + w]

        imgProposition = cv2.resize(roi, compareSize)
        #not necessary anymore as hog method is doing all that
        imgProposition = imgProposition.flatten() / 255.
        # TODO:preprocess hog here xCurrent
        # imgProposition = hogPreprocessing(imgProposition)

        x_train.append(imgProposition)
        y_train.append(yCurrent)

    #we have to train SVM model
    svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
    svm.fit(x_train, y_train)

    return svm, x_train, y_train


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

    charPhotoResized = cv2.resize(charPhoto,compareSize)
    charTruth = charPhotoResized.flatten()/ 255.

    topDistance = float('inf')
    topLetterDigit = None



    for i in range(1,28):

        fullPath, label = getPhotoPath(i);

        img = cv2.imread(fullPath)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #  detect a contour so there is no background stupid match
        contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y + h, x:x + w]

        imgProposition = cv2.resize(roi, compareSize)
        imgProposition = imgProposition.flatten()/255.

        value = euclideanDistance(imgProposition,charTruth)

        if (value < topDistance):
            topLetterDigit = label
            topDistance = value

    return topLetterDigit,topDistance

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
    return fullPath,label

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


#returns an array of Digits/Letters
def processEachDL(image, epsilon,k1,k2,ratioStandard):
    #1

    listaChars = []

    #We convert RGB->Gray
    imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #separatin into a binary image
    image = isodata_thresholding(imgGray,epsilon)
    # inverting so letter/digits are white the plate is black
    imageInverted = 255-image

    # plt.imshow(cv2.cvtColor(imageInverted,cv2.COLOR_GRAY2RGB))
    # plt.title('Green')
    # plt.axis('off')
    # plt.show()

    # Calculate an area of the image so then we can use that to discard not valid contours
    height, width = image.shape[:2]
    MAX_AREA = height*width

    #Now if we have find countours for each digit
    contours,_ = cv2.findContours(imageInverted,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        #if this contour is too big we discard it
        if(MAX_AREA*k1<area or MAX_AREA*k2>area):
            continue


        x, y, w, h = cv2.boundingRect(contour)


        #ratio
        roi = imageInverted[y:y + h, x:x + w]

        nonZeroPixelsInTheRectangle = cv2.countNonZero(roi)
        allPixelsInTheRectangle = w * h
        ratio = nonZeroPixelsInTheRectangle/allPixelsInTheRectangle

        if(ratio<ratioStandard):
            continue



        imageCropped = imageInverted[y:y + h, x:x + w]
        listaChars.append(imageCropped)

        # visualization
        # plt.imshow(cv2.cvtColor(imageCropped, cv2.COLOR_GRAY2RGB))
		#
        # plt.title('Green')
        # plt.axis('off')
        # plt.show()
    return listaChars



def isodata_thresholding(image, epsilon = 2):
    # Compute the histogram and set up variables
    hist = np.array(cv2.calcHist([image], [0], None, [256], [0, 256])).flatten()
    tau = np.random.randint(hist.nonzero()[0][0], 256 - hist[::-1].nonzero()[0][0])
    old_tau = -2*epsilon
    # Iterations of the isodata thresholding algorithm
    while(abs(tau - old_tau) >= epsilon):
        ForegroundMask = image >= tau
        #TODO Calculate m1
        m1 = image[ForegroundMask]
        #this so there is no division by zero problem
        m1Len = np.count_nonzero(m1) if np.count_nonzero(m1)!=0 else 0.0001
        m1Sum = np.sum(m1)
        m1 = m1Sum/m1Len
        #TODO Calculate m2
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
    image = np.where(image>=tau,255,0).astype(np.uint8)
    return image

if __name__ == "__main__":
    plate_image_path = "recognitionTestPlate.png"
    plate_image = cv2.imread(plate_image_path)
    if plate_image is None:
        print(f"Error: Could not load image from path {plate_image_path}")
    else:
        segment_and_recognize(plate_image)