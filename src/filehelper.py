from pydub import AudioSegment
import matplotlib.pyplot as plt
import subprocess
from os import listdir, makedirs, pardir
from os.path import isfile, join, basename, isdir, exists, abspath
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import shutil

def split_file_to_parts(filepath, parts, output_folder):
    sound = AudioSegment.from_file(filepath)
    part_size = int(len(sound) // float(parts))
    file_parts = []

    format_type = filepath.split('.')[-1]
    file_name = basename(filepath).split('.')[0]

    for part_index in range(parts):
        from_index = part_size * part_index
        to_index = (part_size * part_index) + part_size
        
        if(to_index >= len(sound)):
            to_index = (len(sound) - 1)

        file_part = sound[from_index:to_index]

        part_directory = output_folder + file_name + '\\'

        if not exists(part_directory):
            makedirs(part_directory)

        file_part.export(part_directory + file_name + "_" + str(part_index + 1) + "." + format_type, format=format_type)
        file_parts.append(file_part)

    return [join(output_folder, f) for f in listdir(output_folder) if isfile(join(output_folder, f))]

def load_all_files(filepath, output_folder, parts=10):
    audio_files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    audio_parts = []
    for audio_file in audio_files:
        for audio_file_part in split_file_to_parts(audio_file, parts, output_folder):
            audio_parts.append((audio_file_part))

    return audio_parts

def delete_and_create_directory(directory):
    delete_directory(directory)
    if not exists(directory):
        makedirs(directory)

def delete_directory(directory):
    if(isdir(directory)):
        shutil.rmtree(directory)

def convert_files_to_wav(filepath):
    audio_files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    for ff in audio_files:
        command = "ffmpeg -i " + ff + " -ab 160k -ac 1 -ar 16000 -vn " + ff.replace('mp4','wav')
        subprocess.call(command, shell=True)

def load_and_organize_files(input_mp4_directory, parts):
    parent_directory = abspath(join(input_mp4_directory, pardir))
    
    input_directory = parent_directory + '\\wav\\'
    temp_directory = parent_directory + '\\temp\\'

    delete_and_create_directory(temp_directory)
    
    if not exists(input_directory):
        delete_and_create_directory(input_directory)
        convert_files_to_wav(input_mp4_directory)
    
    return load_all_files(input_directory, temp_directory, parts), [abspath(join(temp_directory, name)) for name in listdir(temp_directory)]
