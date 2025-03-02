import cv2
import os

import joblib
import pandas as pd
import Localization
import Recognize
import matplotlib
import matplotlib.pyplot as plt






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

    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
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

    # TODO: Implement actual algorithms for Localizing Plates
    # TODO: UNCOMMENT IF U WANT TO WORK HERE
    count=0
    listaResults = []
    plates = []
    for frameName, frame in frames:
        result = Localization.plate_detection(frame)
        if not result or result[0] is None:
            continue

        plateCropped = result[0].croppedImage
        plateText = Recognize.segment_and_recognize(plateCropped, svm, svm2, scaler)

        plates.append((plateText, frameName, frameName / framesPerSecond))



    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")


    # output.write("XS-NB-23,34,1.822\n")
    for plateText, frameNo, timestamp in plates:
        output.write(f"{plateText},{frameNo}, {timestamp:.3f}\n")

    pass


def showImage(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

class Result():
    def __init__(self, frameNumber, x, y,w,h, timeFrame):
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


