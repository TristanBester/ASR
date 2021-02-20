import os
import re
import string
import argparse
import pandas as pd
import torchaudio
from tqdm import tqdm
from utils import TextEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='root directory of \
                        LibriSpeech dataset.', default='LibriSpeech')
    parser.add_argument('--dataset_name', type=str, help='name of LibriSpeech \
                        subset that is to be preprocessed.', default='train-clean-100')
    parser.add_argument('--resample', type=bool, default=True)
    parser.add_argument('--new_sample_rate', type=int, default=8000)
    parser.add_argument('--remove_old', type=bool, default=True)
    args = parser.parse_args()
    return args

def get_all_paths(root_dir, dataset_name):
    audio_file_paths = []
    text_file_paths = []

    speaker_dir = os.path.join(root_dir, dataset_name)
    for speaker in os.listdir(speaker_dir):
        chapter_dir = os.path.join(speaker_dir, speaker)
        for chapter in os.listdir(chapter_dir):
            utterance_dir = os.path.join(chapter_dir, chapter)
            for file in os.listdir(utterance_dir):
                file_path = os.path.join(utterance_dir, file)
                if file[0] != 'r':
                    if file.split('.')[1] == 'flac':
                        audio_file_paths.append(file_path)
                    else:
                        text_file_paths.append(file_path)
    return audio_file_paths, text_file_paths

def get_all_metadata(text_file_paths):
    all_text = []

    for text_file in text_file_paths:
        with open(text_file) as file:
            all_text += file.readlines()
    return all_text

def split_metadata(raw):
    split_meta = re.split('([0-9]+)|(\n)', raw)
    split_meta = [x for x in split_meta if x is not None and
                  x not in ('', '\n', '-')]
    return split_meta

def clean_utterance(raw):
    raw = raw[1:]
    raw = str.lower(raw)
    raw = raw.translate(str.maketrans('', '', string.punctuation))
    return raw

def get_audio_path(row, root_dir, dataset_name):
    utterance_id = int(row['utterance_id'])
    file_name = 'rs-' + row['reader_id']  + '-' + row['chapter_id'] + '-' + \
                f'{utterance_id:04d}.flac'
    path = os.path.join(root_dir, dataset_name, row['reader_id'],
                        row['chapter_id'], file_name)
    return path

def resample_audio_file(file_path, new_sample_rate):
    waveform, sample_rate = torchaudio.load(file_path)
    transform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
    resampled_wave = transform(waveform)
    split = file_path.split('/')
    file_path = split[:-1] + ['rs-' + str(split[-1])]
    file_path = '/'.join(file_path)
    torchaudio.save(file_path, resampled_wave, new_sample_rate)


if __name__ == '__main__':
    args = get_args()
    encoder = TextEncoder()
    audio_file_paths, text_file_paths = get_all_paths(args.root_dir, args.dataset_name)
    all_metadata = get_all_metadata(text_file_paths)

    # Prepare the CSV file to be used with the dataset.
    print(f'Creating CSV file for {args.dataset_name}')
    df = pd.DataFrame({'raw_metadata':all_metadata})

    df['split_metadata'] = df['raw_metadata'].apply(split_metadata)
    df['len_metadata'] = df['split_metadata'].apply(lambda x: len(x))
    df.drop(df[df['len_metadata'] < 4].index, axis=0, inplace=True)
    df.drop('len_metadata', axis=1, inplace=True)
    df['reader_id'] = df['split_metadata'].apply(lambda x: x[0])
    df['chapter_id'] = df['split_metadata'].apply(lambda x: x[1])
    df['utterance_id'] = df['split_metadata'].apply(lambda x: x[2])
    df['label'] = df['split_metadata'].apply(lambda x: x[3])
    df['label'] = df['label'].apply(clean_utterance)
    df['int_encoded_label'] = df['label'].apply(encoder.char_to_int)
    df['char_encoded_label'] = df['int_encoded_label'].apply(encoder.int_to_char)
    df['file_path'] = df.apply(get_audio_path,  args=(args.root_dir, args.dataset_name), axis=1)

    df.to_csv(f'{args.dataset_name}.csv', index=False)

    # Preprocess the audio files.
    if args.resample:
        print('Resamping audio files.')
        pbar = tqdm(audio_file_paths, leave=True, total=len(audio_file_paths))
        for path in pbar:
            resample_audio_file(path, args.new_sample_rate)

        if args.remove_old:
            print('Removing original audio files.')
            pbar = tqdm(audio_file_paths, leave=True, total=len(audio_file_paths))
            for path in pbar:
                os.remove(path)
