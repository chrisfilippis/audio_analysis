from pydub import AudioSegment
from os.path import isfile, join, basename, isdir, exists, abspath
import matplotlib.pyplot as plt
from os import listdir, makedirs, pardir
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import filehelper
import numpy as np

def clustering(parts):
    train_test_split([], [], test_size=0.4, random_state=0)

def plot_sample(features):
    # path = "C:\ProgramData\Anaconda2\Lib\site-packages\pyAudioAnalysis\data\diarizationExample.wav"
    plt.subplot(2,1,1)
    plt.plot(features[0,:])
    plt.xlabel('Frame no')
    plt.ylabel('ZCR')
    plt.subplot(2,1,2)
    plt.plot(features[1,:])
    plt.xlabel('Frame no')
    plt.ylabel('Energy')
    plt.show()

def find_labels(audio_partfiles, label_index):
    labels = []
    label_index = 1

    for audio_partfile in audio_partfiles:
        parts = basename(audio_partfile).split('.')[-2].split('_')
        label_value = parts[label_index]
        if label_value not in labels:
            labels.append(label_value)

    return labels

def test(audio_partfiles):
    labels = []
    label_index = 1

    for audio_partfile in audio_partfiles:
        parts = basename(audio_partfile).split('.')[-2].split('_')
        label_value = parts[label_index]
        if label_value not in labels:
            labels.append(label_value)

    print labels

    data_X = list()
    data_y = list()

    for audio_partfile in audio_partfiles:
        print audio_partfile
        Fs, x = audioBasicIO.readAudioFile(audio_partfiles[0])
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
        parts = basename(audio_partfile).split('.')[-2].split('_')
        # gender = parts[0]
        # speaker = parts[1]
        # text_index = parts[2]
        # part_index = parts[3]
        label = parts[label_index]
        for ii in F:
            data_X.append(ii)
            data_y.append(labels.index(label))

    data_X = np.array(data_X)

    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3, random_state=5)

    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(train_X, train_y)
    pred_y = neigh.predict(test_X)
    print accuracy_score(test_y, pred_y)

def trim_arrays(dictionary):
    lengths = []
    
    for key in sorted(dictionary.keys()):
        dict_value = dictionary[key]
        print dict_value.shape
        lengths.append(len(dict_value[1]))
    
    max_length = max(lengths)
    print lengths
    print 'max length: '+ str(max_length)


    for key in sorted(dictionary.keys()):
        print key
        dict_value = dictionary[key]
        if dict_value.shape[1] != max_length:
            dim = (max_length - dict_value.shape[1])
            new_data = np.zeros((dict_value.shape[0], dim))
            new_dict_data = np.hstack((dict_value, new_data))
            print new_dict_data.shape
            dictionary[key] = new_dict_data
            print dictionary[key].shape
            # dict_value dict_value[:,0:max_length]
    
    lengths = []

    for ke in sorted(dictionary.keys()):
        print ke
        dict_val = dictionary[ke]
        print dict_val.shape
        
    return dictionary

def knn_classification(audio_partfiles):
    labels = []
    label_index = 1

    labels = find_labels(audio_part_files, label_index)

    data_X = {}
    data_y = list()

    print labels

    for audio_partfile in audio_partfiles[0:150]:
        
        Fs, x = audioBasicIO.readAudioFile(audio_partfile)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
        parts = basename(audio_partfile).split('.')[-2].split('_')

        label = parts[label_index]
        label_ind = str(labels.index(label))

        if(label_ind not in data_X):
            data_X[label_ind] = np.array(F)
        else:
            key_data = data_X[label_ind]
            key_data = np.append(key_data, F, axis=1)
            data_X[label_ind] = key_data
        
    data_X = trim_arrays(data_X)

    data = []
    for key in sorted(data_X.keys()):
        data.append(np.array(data_X[key]))

    print np.array(data).shape
    exit()
    
    # for i in data:
    #     for j in i:
    #         print type(j)
    #         print len(j)

    classifier = aT.trainKNNFalse(data, 6)
    print classifier

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

audio_part_files, directories = filehelper.load_file_data('C:\Users\\filippisc\Projects\Master\\audio_analysis\data\mp4\\', 10)
audio_part_files = np.array(audio_part_files)

# knn_classification(audio_part_files)

testing_files = get_test_files(directories, '_1')

aT.featureAndTrain(filter_directories(directories, '_0'), 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", False)

for testing_file in testing_files[0:30]:
    print testing_file
    print aT.fileClassification(testing_file, "knnMusicGenre3", "knn")

# aT.featureAndTrain(directories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", False)
# test(audio_part_files)
# plot_sample(F)
# clustering(audio_partfiles)

