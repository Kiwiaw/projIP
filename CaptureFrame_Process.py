from itertools import product

import cv2
import os

import joblib
import pandas as pd
import Localization
import Recognize
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from main import addDutchDashes, makeDucthPlate, isDutchPlate


def CaptureFrame_Process(file_path, sample_frequency, save_path):

    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """

#TODO: Read frames from the video (saved at file_path) by making use of sample_frequency
    frames = []
    cam = cv2.VideoCapture(file_path)
    frameName = 0
    framesPerSecond = cam.get(cv2.CAP_PROP_FPS)

    interval  =  int(sample_frequency*framesPerSecond)
    flag = True
    while (flag):
        cam.set(cv2.CAP_PROP_POS_FRAMES, frameName)
        ret, frame = cam.read()

        if ret:
            name = str(frameName) + '.jpg'
            print('photo Number: ' + name)


            frames.append((frameName,frame))
            frameName += interval
        else:
            flag = False


    cam.release()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(frames)



    svm, svm2, scaler = joblib.load('svm2_model.pkl'), joblib.load('svm2_model.pkl'), joblib.load('scaler.pkl')
    print('Models loaded !<3')

    #TODO: Implement actual algorithms for Localizing Plates
    #TODO: UNCOMMENT IF U WANT TO WORK HERE
    count=0
    listaResults = []
    plates = []
    previousPlate = None
    plateTexts = []
    previousText = ""
    timestamp = 0

    for frameName, frame in frames:
        result = Localization.plate_detection(frame)
        if not result or result[0] is None:
            continue

        candidates = []
        for contour in result:
            candidate = Recognize.segment_and_recognize(contour.croppedImage, svm, svm2, scaler)
            if len(candidate) == 6:
                candidates.append((contour, candidate))
        if candidates:
            found = False
            if(len(candidates) > 1):
                for contour, candidate in candidates:
                    if isDutchPlate(candidate):
                        currentPlate = contour
                        currentText = candidate
                        found = True
                        break

            if(not found):
                currentPlate = candidates[0][0]
                currentText = candidates[0][1]
        else:
            currentPlate = result[0]
            currentText = Recognize.segment_and_recognize(contour.croppedImage, svm, svm2, scaler)

        # plt.title("current")
        # showImage(currentPlate.croppedImage)
        # if previousPlate:
            # plt.title("previous")
            # showImage(previousPlate.croppedImage)

        if samePlate(currentPlate, previousPlate, currentText, previousText):
            plateTexts.append(currentText)
        else:
            plateText = majorityVoting(plateTexts)
            plateTexts = [currentText]
            if plateText:
                plateText = makeDucthPlate(plateText)
                plates.append((addDutchDashes(plateText), timestamp, timestamp / framesPerSecond))

            timestamp = frameName

        previousPlate = currentPlate
        previousText = currentText


    # output = open(save_path, "w")
    # output.write("License plate,Frame no.,Timestamp(seconds)\n")

    plateText = majorityVoting(plateTexts)
    if plateText:
        plateText = addDutchDashes(makeDucthPlate(plateText))
        plates.append((plateText, timestamp, timestamp / framesPerSecond))

    plates = removeDuplicates(plates)

    with open(save_path, "w") as output:
        output.write("License plate,Frame no.,Timestamp(seconds)\n")
        for plateText, frameNo, timestamp in plates:
            output.write(f"{plateText},{frameNo},{timestamp:.3f}\n")


# plateLength is the length of the text  of the plate
def majorityVoting(strings, plateLength=6):
    if not strings:
        return ""

    strings = [s for s in strings if len(s) == plateLength]
    if not strings:
        return ""

    length = 6

    allCandidates = []
    for i in range(length):
        charCounts = Counter(s[i] for s in strings)
        totalVotes = sum(charCounts.values())

        candidates = [char for char, count in charCounts.items() if count / totalVotes >= 0.2]

        if not candidates:
            candidates.append(max(charCounts, key=charCounts.get))

        allCandidates.append(candidates)

    for combination in product(*allCandidates):
        plate = "".join(combination)
        if isDutchPlate(plate):
            return plate
    return "".join(max(Counter(s[i] for s in strings), key=Counter(s[i] for s in strings).get) for i in range(length))

def haveCommonChars(s1: str, s2: str, count=3) -> bool:
    difference_count = defaultdict(int)

    char_indices_str2 = defaultdict(list)
    for idx, char in enumerate(s2):
        char_indices_str2[char].append(idx)

    for i, char in enumerate(s1):
        if char in char_indices_str2:
            for j in char_indices_str2[char]:
                diff = i - j
                difference_count[diff] += 1

                if difference_count[diff] >= count:
                    return True

    return False

def samePlate(current, previous, currentText, previousText):
    if previous is None:
        return False

    def intersectionOverBiggerArea(x1, y1, w1, h1, x2, y2, w2, h2, commonArea=0.70):
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        intersection = x_overlap * y_overlap
        biggerArea = max(w1 * h1, w2 * h2)
        return intersection / biggerArea >= commonArea


    return (haveCommonChars(currentText, previousText) or
            intersectionOverBiggerArea(current.x, current.y, current.w, current.h, previous.x, previous.y, previous.w, previous.h))

def removeDuplicates(strings: list[str], count=4) -> list[str]:
    if not strings:
        return []

    result = [strings[0]]

    for i in range(1, len(strings)):
        if not haveCommonChars(strings[i][0], strings[i - 1][0], count=count):
            result.append(strings[i])

    return result

def showImage(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

class Result():
    def _init_(self, frameNumber, x, y,w,h, timeFrame):
        self.frameNumber = frameNumber
        self.x =x
        self.y =y
        self.w = w
        self.h = h
        self.timeFrame = timeFrame

    def getFrameNumber(self):
        return self.frameNumber

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

    def getTimeFrame(self):
        return self.timeFrame