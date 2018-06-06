from pydub import AudioSegment
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import filehelper

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

audio_partfiles = filehelper.load_file_data('C:\Users\\filippisc\Projects\Master\\audio_analysis\data\mp4\\', 10)

print audio_partfiles[0]
Fs, x = audioBasicIO.readAudioFile(audio_partfiles[0])
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
plot_sample(F)
# clustering(audio_partfiles)

