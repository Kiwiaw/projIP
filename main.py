import argparse
import os

import cv2
from matplotlib import pyplot as plt

import CaptureFrame_Process
import numpy
import pandas as pd

# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
import Localization
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/trainingvideo.avi')
    parser.add_argument('--output_path', type=str, default='dataset/Output.csv')
    parser.add_argument('--sample_frequency', type=int, default=2)
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
# CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path)
#


def groundTruthRead():
    pathGT = "dataset/groundTruth.csv"
    GT = pd.read_csv(pathGT)
    print(GT.head())


# groundTruthRead()

# we just run it for every frame between the start and end of the GT frames
# and then take the higehest accuracy
def checkAccuracyIOU(x1, y1, w1, h1, x2, y2, w2, h2):

    # intersection/union
    # x,y,w,h predictions and GT

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
    # startFrame-=24;
    # endFrame-=24;

    framesPerSecond = cam.get(cv2.CAP_PROP_FPS)

    # adding to frames all the frames between startFrame and endFrame
    #We can change so the accuracy is better !!! (these 12 to 1)
    for currentFrame in range(startFrame, endFrame + 1, 12):
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
        # cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.title('Red')
        # plt.axis('off')
        # plt.show()
        result = Localization.plate_detection(frame)
        if result[0] is None:
            continue
        image, x, y, w, h = result
        #looking for bugs - visualization green rectagle
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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




    listaResults,debugDisctionary = fetchStartEndGTFrameValues(file_path, startFrame, endFrame,x1, y1, w1, h1)
    biggestAccuracy = 0
    lastResult = 0
    for R in listaResults:
        print(f'x:{R.x}, y:{R.y}, w:{R.w} and h: {R.h} ')
        print(f'GR: x:{x1}, y:{y1}, w:{w1} and h: {h1} ')
        currentAccuracy = checkAccuracyIOU(x1, y1, w1, h1, R.x, R.y, R.w, R.h)

        print(f'frame: {R.frameNumber}, currentAccuracy: {currentAccuracy}')
        if (currentAccuracy > biggestAccuracy):
            biggestAccuracy = currentAccuracy
            lastResult =R

    image = debugDisctionary[R]
    # check, from ground truth
    # cv2.rectangle(image, (R.x, R.y), (R.x + R.w, R.y + R.h), (0, 0, 255), 2)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Red')
    # plt.axis('off')
    # plt.show()
    #
    # cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Green - it should be like this')
    # plt.axis('off')
    # plt.show()

    return biggestAccuracy


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
            file_path = os.path.join(category_path, file_name)
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
                accuracy = getHighestAccuracyForEachOneGT(int(left), int(top), int(width), int(height), start , end, 'dataset/trainingvideo.avi')
    return accuracy

# Accuracy for Cat1 Train
def AccuracyForFullSet(Category):
    sumAccuracy=-1
    for frame in Category:
        sumAccuracy+=processJsonGetAccuracy(frame)

    return sumAccuracy/len(Category)
basePath = "dataset/GT Train/CAT 2"
# print(f'Average accuarcy: {AccuracyForFullSet(Cat2Train)}')
#
