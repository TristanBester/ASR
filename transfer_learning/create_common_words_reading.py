import random

with open('common_word_list.txt', 'r') as f:
    word_ls = f.readlines()

word_ls = [i.replace('\n','') for i in word_ls]

lines = []
for i in range(36):
    line = random.sample(word_ls, 10)
    lines.append(' '.join(line))

with open('common_words_reading.txt', 'w') as f:
    for i in lines:
        f.write(i + '\n')
