import subprocess
from os import listdir
from os.path import isfile, join, basename


def convert_files_to_wav(filepath):
    audio_files = [join(filepath, f) for f in listdir(filepath) if isfile(join(filepath, f))]
    for ff in audio_files:
        command = "ffmpeg -i " + ff + " -ab 160k -ac 2 -ar 44100 -vn " + ff.replace('mp4','wav')
        subprocess.call(command, shell=True)

filepath = "C:/Projects/Master/audio_analysis/data/mp4"
convert_files_to_wav(filepath)
