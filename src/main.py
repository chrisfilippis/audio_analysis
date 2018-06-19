from pydub import AudioSegment
from os.path import isfile, join, basename, isdir, exists, abspath, dirname
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
import itertools

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate(testingfiles):
    actual_y = []
    predict_y = []

    for testing_file in testingfiles:
        prediction_result = aT.fileClassification(testing_file, "knnMusicGenre3", "knn")
        prediction = prediction_result[2][prediction_result[0]]
        actual_y.append(path_leaf(testing_file).split('_')[1])
        predict_y.append(prediction.split('_')[1])
    
    print accuracy_score(actual_y, predict_y)
    conf_matrix = confusion_matrix(actual_y, predict_y)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(actual_y), normalize=False, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=set(actual_y), normalize=True, title='Normalized confusion matrix')
    plt.show()

def train_and_evaluate(train_index, test_index, init_path, parts):
    relative_path = abspath(dirname(__file__))
    directory_path = join(relative_path, init_path)
    a_f, directories = filehelper.load_and_organize_files(directory_path, parts)
    testing_files = get_test_files(directories, '_' + str(train_index))

    aT.featureAndTrain(filter_directories(directories, '_' + str(test_index)), 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", False)

    evaluate(testing_files)

train_audio_index = '0'
test_audio_index = '1'
audiofiles_directory = '..\data\mp4\\'
audio_parts = 10
train_and_evaluate(train_audio_index, test_audio_index, audiofiles_directory, audio_parts)