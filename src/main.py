from pydub import AudioSegment
from os.path import isfile, join, basename, isdir, exists, abspath
import matplotlib.pyplot as plt
from os import listdir, makedirs, pardir
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import filehelper
import numpy as np
import ntpath

def filter_directories(directories, pattern):
    result = []
    for directory in directories:
        if(directory.endswith(pattern)):
            result.append(directory)
    
    return result

def get_test_files(directories, pattern):
    f_directories = filter_directories(directories, pattern)
    result = []
    for directory in f_directories:
        for f in listdir(directory):
            if isfile(join(directory, f)):
                result.append(join(directory, f))

    return result

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def evaluate(testingfiles):
    actual_y = []
    predict_y = []

    for testing_file in testingfiles:
        prediction_result = aT.fileClassification(testing_file, "knnMusicGenre3", "knn")
        prediction = prediction_result[2][prediction_result[0]]
        actual_y.append(path_leaf(testing_file).split('_')[1])
        predict_y.append(prediction.split('_')[1])
    
    return actual_y, predict_y

a_f, directories = filehelper.load_and_organize_files('C:\Users\\filippisc\Projects\Master\\audio_analysis\data\mp4\\', 10)
testing_files = get_test_files(directories, '_1')

aT.featureAndTrain(filter_directories(directories, '_0'), 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", False)

y_true, y_pred = evaluate(testing_files[0:130])

print accuracy_score(y_true, y_pred)
print confusion_matrix(y_true, y_pred)