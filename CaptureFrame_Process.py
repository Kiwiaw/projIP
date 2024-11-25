import cv2
import os
import pandas as pd
import Localization
import Recognize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


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
        ret, frame = cam.read()
        if ret:


            name = str(frameName) + '.jpg'
            print('photo Number: ' + name)

            frameName += interval
            frames.append(frame)
        else:
            flag = False


    cam.release()
    cv2.destroyAllWindows()
    print(frames)

    # TODO: Implement actual algorithms for Localizing Plates
    for frame in frames:
        # Localization.plate_detection(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.show()

        print(type(frame), frame.shape)

        break

    # TODO: Implement actual algorithms for Recognizing Characters

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    output.write("XS-NB-23,34,1.822\n")
    # output.write("YOUR,STUFF,HERE\n")
    # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass
