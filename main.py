import argparse
import os

import cv2

import CaptureFrame_Process
import numpy
import pandas as pd

# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
import Localization


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


def fetchStartEndGTFrameValues(file_path, startFrame, endFrame):
    frames = []
    cam = cv2.VideoCapture(file_path)

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

    for frameIndex, (frameName, frame) in enumerate(frames):
        result = Localization.plate_detection(frame)
        if result[0] is None:
            continue
        image, x, y, w, h = Localization.plate_detection(frame)
        if image is None or image.size == 0:
            continue
        listaResults.append(CaptureFrame_Process.Result(frameName, x, y, w, h, frameName / framesPerSecond))

    # return list with Result
    return listaResults


def getHighestAccuracyForEachOneGT(x1, y1, w1, h1, startFrame, endFrame, file_path):
    # timeframe from the video  - 24 (gives a good result)
    # TODO: we have to fix it ^^^
    # startFrame -= 24
    # endFrame -= 24
    listaResults = fetchStartEndGTFrameValues(file_path, startFrame, endFrame)
    biggestAccuracy = 0
    for R in listaResults:
        print(f'x:{R.x}, y:{R.y}, w:{R.w} and h: {R.h} ')
        print(f'GR: x:{x1}, y:{y1}, w:{w1} and h: {h1} ')
        currentAccuracy = checkAccuracyIOU(x1, y1, w1, h1, R.x, R.y, R.w, R.h)

        print(f'frame: {R.frameNumber}, currentAccuracy: {currentAccuracy}')
        if (currentAccuracy > biggestAccuracy):
            biggestAccuracy = currentAccuracy
    return biggestAccuracy


# TEST
# timeframe from the video  - 24 (gives a good result)
# TODO: we have to fix it ^^ >_<
accuracyCheck = getHighestAccuracyForEachOneGT(307, 290, 225, 53, 90*12, 95*12, 'dataset/trainingvideo.avi')
print(accuracyCheck)





def divideIntoSets():
    Cat1 = []
    Cat2 = []
    Cat3 = []
    Cat4 = []

    basePath = r"dataset/TrainingSet"

    categories = {
        "Categorie I": Cat1,
        "Categorie II": Cat2,
        "Categorie III": Cat3,
        "Categorie IV": Cat4,
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
