import argparse
import csv
import os

import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt


import numpy
import pandas as pd

# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
import CaptureFrame_Process
import Localization
import json

import Recognize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/trainingvideo.avi')
    parser.add_argument('--output_path', type=str, default='dataset/Output.csv')
    parser.add_argument('--sample_frequency', type=int, default=1/2)
    args = parser.parse_args()
    return args


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
    args = get_args()
    if args.output_path is None:
        output_path = os.getcwd()
    else:
        output_path = args.output_path
    file_path = args.file_path
    sample_frequency = args.sample_frequency
    basePath = "dataset/GT Train/CAT 1"
    CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path)
#




# we just run it for every frame between the start and end of the GT frames
# and then take the higehest accuracy
def checkAccuracyIOU(x1, y1, w1, h1, x2, y2, w2, h2):

    # intersection/union
    # x,y,w,h predictions and GT
    wTolerance = int(0.1 * w1)
    hTolerance = int(0.05 * h1)

    if (x2 >= x1 - wTolerance and
            y2 >= y1 - hTolerance and
            x2 + w2 <= x1 + w1 + wTolerance and
            y2 + h2 <= y1 + h1 + hTolerance):
        return 1.0

    coordiatesImage1 = [(i, j) for j in range(y1, y1 + h1) for i in range(x1, x1 + w1)]
    coordiatesImage1 = set(coordiatesImage1)
    coordiatesImage2 = [(i, j) for j in range(y2, y2 + h2) for i in range(x2, x2 + w2)]
    coordiatesImage2 = set(coordiatesImage2)

    # now do
    intersection = coordiatesImage1.intersection(coordiatesImage2)
    intersectionLen = len(intersection)

    union = h1 * w1 + h2 * w2 - intersectionLen


    # intersectionLen/union

    IOU = intersectionLen / union

    return IOU


def fetchStartEndGTFrameValues(file_path, startFrame, endFrame,x1, y1, w1, h1):
    frames = []
    cam = cv2.VideoCapture(file_path)


    framesPerSecond = cam.get(cv2.CAP_PROP_FPS)

    # adding to frames all the frames between startFrame and endFrame
    #We can change so the accuracy is better !!! (these 12 to 1)
    for currentFrame in range(startFrame, endFrame + 1, 2):
        cam.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
        ret, frame = cam.read()
        if ret:
            name = str(currentFrame) + '.jpg'
            print('photo Number: ' + name)
            frames.append((currentFrame, frame))
        else:
            break

    cam.release()

    count = 0
    # lista with result Result
    listaResults = []

    #result accuracy, image
    debugDisctionary ={}


    for frameIndex, (frameName, frame) in enumerate(frames):
        # check, from ground truth
        unmodified_frame = frame.copy()
        # cv2.rectangle(unmodified_frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        # plt.imshow(cv2.cvtColor(unmodified_frame, cv2.COLOR_BGR2RGB))
        # plt.title('Red')
        # plt.axis('off')
        # plt.show()
        result = Localization.plate_detection(frame)
        for r in result:

            if r.croppedImage is None:
                continue
            image, x, y, w, h = r.croppedImage,r.x,r.y,r.w,r.h


            #looking for bugs - visualization green rectagle
            # cv2.rectangle(unmodified_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # plt.imshow(cv2.cvtColor(unmodified_frame, cv2.COLOR_BGR2RGB))
            # plt.title('Green')
            # plt.axis('off')
            # plt.show()
            #
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
            # plt.title('Green')
            # plt.axis('off')
            # plt.show()


            if image is None or image.size == 0:
                continue
            resultHere = CaptureFrame_Process.Result(frameName, x, y, w, h, frameName / framesPerSecond)
            listaResults.append(resultHere)
            debugDisctionary[resultHere] =frame
    # return list with Result

    return listaResults,debugDisctionary


def getHighestAccuracyForEachOneGT(x1, y1, w1, h1, startFrame, endFrame, file_path):
    print("Entering getHighestAccuracyForEachOneGT")

    listaResults, debugDisctionary = fetchStartEndGTFrameValues(file_path, startFrame, endFrame, x1, y1, w1, h1)
    print(f"Length of listaResults: {len(listaResults)}")

    listaResults,debugDisctionary = fetchStartEndGTFrameValues(file_path, startFrame, endFrame,x1, y1, w1, h1)
    biggestAccuracy = 0
    lastResult = None
    for R in listaResults:
        print(f'x:{R.x}, y:{R.y}, w:{R.w} and h: {R.h} ')
        print(f'GR: x:{x1}, y:{y1}, w:{w1} and h: {h1} ')
        currentAccuracy = checkAccuracyIOU(x1, y1, w1, h1, R.x, R.y, R.w, R.h)

        print(f'frame: {R.frameNumber}, currentAccuracy: {currentAccuracy}')
        if (currentAccuracy > biggestAccuracy):
            biggestAccuracy = currentAccuracy
            lastResult =R
    image = debugDisctionary.get(lastResult, None)
    #best detected squares for all
    # x,y,w,h = lastResult.x,lastResult.y,lastResult.w, lastResult.h
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected best square over plate')
    # plt.axis('off')
    # plt.show()

    #Here can be called image recognition function for characters
    #TODO: problem with None, returns= NOne for some photos
    print(f'Higest accuracy is: {biggestAccuracy}')
    if image is None:
        return biggestAccuracy, None, lastResult
    return biggestAccuracy, image,lastResult


# TEST

# accuracyCheck = getHighestAccuracyForEachOneGT(307, 290, 225, 53, 90*12, 95*12, 'dataset/trainingvideo.avi')
# print(accuracyCheck)





def divideIntoSets():
    Cat1 = []
    Cat2 = []
    Cat3 = []
    Cat4 = []

    basePath = r"dataset/GT Train"

    categories = {
        "CAT 1": Cat1,
        "CAT 2": Cat2,
        "CAT 3": Cat3,
        "CAT 4": Cat4,
    }

    for category, file_list in categories.items():
        category_path = os.path.join(basePath, category)
        all_entries = os.listdir(category_path)
        for file_name in all_entries:
            file_list.append(file_name)

    splitFactor = 0.7

    Cat1Train = Cat1[:int(splitFactor * len(Cat1))]
    Cat1Test = Cat1[int(splitFactor * len(Cat1)):]

    Cat2Train = Cat2[:int(splitFactor * len(Cat2))]
    Cat2Test = Cat2[int(splitFactor * len(Cat2)):]

    Cat3Train = Cat3[:int(splitFactor * len(Cat3))]
    Cat3Test = Cat3[int(splitFactor * len(Cat3)):]

    Cat4Train = Cat4[:int(splitFactor * len(Cat4))]
    Cat4Test = Cat4[int(splitFactor * len(Cat4)):]
    return Cat1Train, Cat1Test, Cat2Train, Cat2Test,Cat3Train, Cat3Test,Cat4Train, Cat4Test

Cat1Train, Cat1Test, Cat2Train, Cat2Test,Cat3Train, Cat3Test,Cat4Train, Cat4Test=divideIntoSets()

#chg
def processJsonGetAccuracy(filePath):


    with open(os.path.join(basePath, filePath), 'r') as f:
        data = json.load(f)
    accuracy =0
    image = None
    lastResult = None
    if 'asset' in data and 'regions' in data:
        timestamp = data["asset"].get("timestamp", 0)

        for region in data["regions"]:
            if 'boundingBox' in region:
                bounding_box = region["boundingBox"]
                left = bounding_box["left"]
                top = bounding_box["top"]
                width = bounding_box["width"]
                height = bounding_box["height"]
                #can be changed later
                start = timestamp - 2
                end= timestamp + 2
                start*=12
                end*=12

                #path can be chnaged later
                accuracy, image,lastResult = getHighestAccuracyForEachOneGT(int(left), int(top), int(width), int(height), start , end, 'dataset/trainingvideo.avi')
    return accuracy,image,lastResult

def addDutchDashes(plate):
    consecutive = 0
    dashIndices = []

    for i, char in enumerate(plate):
        if i > 0:
            if char.isdigit() and plate[i - 1].isalpha()\
                    or char.isalpha() and plate[i - 1].isdigit():
                consecutive = 0
                dashIndices.append(i)

        if char.isalpha():
            consecutive += 1
            if consecutive == 4:
                dashIndices.append(i-1)
                consecutive = 2
        else:
            consecutive = 0

    if 2 in dashIndices:
        if 1 in dashIndices:
            dashIndices.remove(1)
        if 3 in dashIndices:
            dashIndices.remove(3)

    if 4 in dashIndices and 5 in dashIndices:
            dashIndices.remove(5)

    if len(dashIndices) == 1:
        if 1 not in dashIndices and 2 not in dashIndices and 3 not in dashIndices:
            dashIndices.append(2)
        elif 1 in dashIndices or 2 in dashIndices:
            dashIndices.append(4)
        else:
            dashIndices.append(5)

    elif len(dashIndices) == 0:
        dashIndices.append(2)
        dashIndices.append(4)

    result = plate
    dashIndices.sort()
    for index in reversed(dashIndices):
        result = result[:index] + "-" + result[index:]
    return result

# Accuracy for Cat1 Train
def AccuracyForFullSet(Category):
    sumAccuracy=0

    # svm, x_train, y_train, scaler = Recognize.svmModel()
    # svm2, _, _ = Recognize.svmModel(compareSize=(64, 64), n_neighbors=3, knnDistance=True, useHog=False)
    svm, svm2, scaler = joblib.load('svm_model.pkl'), joblib.load('svm2_model.pkl'), joblib.load('scaler.pkl')

    #list of plates (expected chars, actual chars, time frame)
    listaExpectedActual = []

    accuracyRecognition = []

    for fileName in Category:
        accuracy, image,lastResult = processJsonGetAccuracy(fileName)
        sumAccuracy+= accuracy
        if(image is None):
            continue
        x,y,w,h = lastResult.x,lastResult.y,lastResult.w, lastResult.h
        showRectangleOnImage(image,x,y,w,h)


        showPlate(image,x,y,w,h)


        basePath = "dataset/groundTruth_platesFileNames.csv"
        plateStringActual, expectedValue  = plateFullExtraction(image[y:y + h, x:x + w], fileName, basePath,svm,svm2,scaler)

        plateStringActual = addDutchDashes(plateStringActual)
        print(f'True value: {expectedValue}, Extracted value: {plateStringActual}')

        currentRecognitionAccuracy = stringAccuracy(expectedValue,plateStringActual)
        accuracyRecognition.append(currentRecognitionAccuracy)
        if(currentRecognitionAccuracy==0.0):
            print("ZERO ACCURACY")
        else:
            print(currentRecognitionAccuracy)

    finalAccuarcy = np.sum(accuracyRecognition)/len(accuracyRecognition)
    print('ACCURACY!!!!!!!!')
    print(finalAccuarcy)


    return sumAccuracy/len(Category)

from collections import Counter

def stringAccuracy(string1, string2):
    if not isinstance(string1, str):
        string1 = str(string1)
    if not isinstance(string2, str):
        string2 = str(string2)

    length = len(string1)

    if length != len(string2) or len == 0:
        return 0.0

    correct_matches = sum(1 for i in range(len(string1)) if string1[i] == string2[i])

    accuracy = (correct_matches / length) * 100
    return accuracy


def plateFullExtraction(image,fileName,basePath,svm,svm2,scaler):
    #actual value
    plateStringActual =  Recognize.segment_and_recognize(image,svm,svm2,scaler)

    #expected value
    expectedValue = ""
    with open(basePath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            listOfFiles = row['file name (plate)'].replace('"', '').split(', ')
            if fileName in listOfFiles:
                expectedValue = row['License plate']
                break
    if not isinstance(expectedValue, str):
        expectedValue = str(expectedValue if expectedValue is not None else "")

    print(f"Expected Value: {expectedValue}, Type: {type(expectedValue)}")
    print(f"Plate String Actual: {plateStringActual}, Type: {type(plateStringActual)}")

    return plateStringActual,expectedValue





def showRectangleOnImage(image, x,y,w,h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #TODO: uncomment
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected best square over plate')
    # plt.axis('off')
    # plt.show()


def showPlate(image,x,y,w,h):
    result = image[y:y + h, x:x + w]
    # TODO: uncomment
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title('detected plate')
    # plt.axis('off')
    # plt.show()


class plateExpectedActual():
    def __init__(self, expected, actual, fileName):
        self.expected = expected
        self.actual = actual
        self.fileName = fileName



# if __name__ == "__main__":
#     basePath = "dataset/GT Train/CAT 1"
#     print(f'Average accuarcy: {AccuracyForFullSet(Cat1Train)}')
# #


