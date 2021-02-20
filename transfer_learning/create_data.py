import queue
import sys
import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write
import torchaudio

def get_sample_rate():
    device_info = sd.query_devices(None, 'input')
    sample_rate = device_info['default_samplerate']
    return int(sample_rate)

def load_labels(path):
    with open(path, 'r') as f:
        labels = f.readlines()
    return labels

def stream_callback(indata, frames, time, status):
    if status:
        print(status)

    global q
    global DOWNSAMPLE
    try:
        q.put(indata[::DOWNSAMPLE, 0].astype(object))
    except:
        print('here')


if __name__ == '__main__':
    WINDOW = 200
    INTERVAL = 30
    DOWNSAMPLE = 1
    CHANNELS = 1
    SAMPLE_RATE = get_sample_rate()
    #SAMPLE_RATE = 8000
    labels = load_labels('difficult_words.txt')
    q = queue.Queue()
    waveforms = []

    i = 0
    start_index = int(input('Enter start index'))

    for label in labels:
        if i < start_index:
            i += 1
            continue

        issue = True
        while issue:
            os.system('cls' if os.name == 'nt' else 'clear')
            print('Please read the following:\n\n')
            print('\x1b[1;36;40m' + label + '\x1b[0m')
            input('Press any key to START the recording.')

            stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE,
                                    callback=stream_callback)
            try:
                with stream:
                    print('Get ready...')
                    time.sleep(3)
                    print('Recording started...')
                    input('Press any key to STOP the recording.')
                    print('dene')
                    time.sleep(3)
            except:
                print('here')
                break

            streamed_data = []
            while not q.empty():
                streamed_data.append(q.get())

            streamed_data = np.array(streamed_data, dtype='object')
            streamed_data = np.hstack(streamed_data).astype(np.float32)[3*SAMPLE_RATE:-int(SAMPLE_RATE*3.05)]

            issue = int(input('Enter 0 if error occurred else 1.\n')) == 0

            if not issue:
                streamed_data /= streamed_data.max()
                write(f'difficult_clips_0/clip{i}.wav', SAMPLE_RATE//DOWNSAMPLE, streamed_data)
        i += 1






################################
