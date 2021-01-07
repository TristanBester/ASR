import numpy as np
import librosa

def get_wav_length_ms(path):
    duration = librosa.get_duration(filename=path)
    duration *= 1000
    return duration

def get_num_samples(time, win_length, overlap):
    eff_win_len = win_length - (overlap * win_length)
    num_samples = np.ceil((time + 0.001) / eff_win_len)
    return num_samples
