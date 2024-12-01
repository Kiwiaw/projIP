import cv2
import os
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

            frameName += interval
            frames.append((frameName,frame))
        else:
            flag = False


    cam.release()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(frames)

    # TODO: Implement actual algorithms for Localizing Plates
    count=0
    listaResults = []
    for frameName, frame in frames:
        image, x,y,w,h=Localization.plate_detection(frame)
        listaResults.append(Result(frameName,x,y,w,h, frameName/framesPerSecond))
        # cv2.imshow("Image After Crop", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        print(f' Time Stamp (in seconds):{listaResults[count].timeFrame}, x:{listaResults[count].x}, '
              f'y:{listaResults[count].y}, w:{listaResults[count].w}, h:{listaResults[count].h} '
              f'and last but not least frame: {listaResults[count].frameNumber}')

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame)
        # plt.show()

        # print(type(frame), frame.shape)


        count+=1
        if (count==10):
            break

    # TODO: Implement actual algorithms for Recognizing Characters

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    output.write("XS-NB-23,34,1.822\n")
    # output.write("YOUR,STUFF,HERE\n")
    # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass

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