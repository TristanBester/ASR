from itertools import product
import os
import re
import pickle

bigram_count = 0
chars = [chr(i) for i in range(97, 123)] + [' ']
bigram_counter = {''.join(i):0 for i in product(chars, repeat=2)}

data_root = 'train-clean-360'
dataset_one_path = os.path.join(data_root, 'train-clean-360-1')
dataset_two_path = os.path.join(data_root, 'train-clean-360-2')

for dataset_path in (dataset_one_path, dataset_two_path):
    for reader in os.listdir(dataset_path):
        reader_path = os.path.join(dataset_path, reader)
        for chapter in os.listdir(reader_path):
            chapter_path = os.path.join(reader_path, chapter)
            for file in os.listdir(chapter_path):
                if file.split('.')[-1] == 'txt':
                    with open(os.path.join(chapter_path, file), 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.lower()
                        line = re.findall('\d+-\d+-\d+|[\w ]+', line)[1][1:]

                        for i in range(len(line)-1):
                            bigram = line[i:i+2]
                            bigram_counter[bigram] += 1
                            bigram_count += 1


for bigram in bigram_counter:
    bigram_counter[bigram] /= bigram_count

with open('bigram-LM.pkl', 'wb') as f:
    pickle.dump(bigram_counter, f)
