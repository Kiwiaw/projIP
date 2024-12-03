import argparse
import os
import CaptureFrame_Process
import numpy
import pandas as pd


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
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
    CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path)
#


def groundTruthRead():
    pathGT = "dataset/groundTruth.csv"
    GT = pd.read_csv(pathGT)
    print(GT.head())


groundTruthRead()


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
