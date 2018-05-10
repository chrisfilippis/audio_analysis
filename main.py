from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

def split_file_to_parts(filepath, parts, output_folder):
    sound = AudioSegment.from_file(filepath)
    part_size = int(len(sound) // float(parts))
    file_parts = []

    format_type = filepath.split('.')[-1]

    for part_index in range(parts):
        from_index = part_size * part_index
        to_index = (part_size * part_index) + part_size
        
        if(to_index >= len(sound)):
            to_index = (len(sound) - 1)

        file_part = sound[from_index:to_index]
        file_part.export(output_folder + "first_half" + str(part_index + 1) + "." + format_type, format=format_type)
        file_parts.append(file_part)
    
    return [join(output_folder, f) for f in listdir(output_folder) if isfile(join(output_folder, f))]

def load_all_files(filepath, parts=10):
    folder = ''
    temp_folder = "C:\Projects\Master\\audio_analysis\data\\test\\"
    audio_files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    audio_parts = []
    for audio_file in audio_files:
        for audio_file_part in split_file_to_parts(audio_file, parts, temp_folder):
            audio_parts.append(audio_file_part)
    return audio_parts

def clustering(parts):
    ok = 1

audio_part_files = load_all_files('C:\ProgramData\Anaconda2\Lib\site-packages\pyAudioAnalysis\data\\', 10)
clustering(audio_part_files)

Fs, x = audioBasicIO.readAudioFile("C:\ProgramData\Anaconda2\Lib\site-packages\pyAudioAnalysis\data\diarizationExample.wav")
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
plt.subplot(2,1,1)
plt.plot(F[0,:])
plt.xlabel('Frame no')
plt.ylabel('ZCR')
plt.subplot(2,1,2)
plt.plot(F[1,:])
plt.xlabel('Frame no')
plt.ylabel('Energy')
plt.show()