import numpy as np

class TextEncoder():
    def __init__(self, blank_first=True):
        idx_shift = 96 if blank_first else 97
        self.char_int_mapping = {chr(i):(i-idx_shift) for i in range(97, 123)}
        self.char_int_mapping[' '] = 27 if blank_first else 26
        self.char_int_mapping['\''] = 28
        self.int_char_mapping = {x:i for i,x in self.char_int_mapping.items()}

    def char_to_int(self, x):
        if x is not None:
            return [self.char_int_mapping[ch] for ch in x]
        else:
            return []

    def int_to_char(self, x):
        if x is not None:
            return [self.int_char_mapping[i] for i in x]
        else:
            return []

class Decoder():
    def __init__(self, blank_val=0):
        self.text_encoder = TextEncoder()
        self.blank_val = blank_val

    def __strip_padding(self, x, pad_val=0):
        if x[-1] != pad_val:
            return x

        for i in range(0, len(x)):
            if x[-i-1] != pad_val:
                return x[:-i]

    def decode_label(self, label, pad_val=0):
        label = self.__strip_padding(label, pad_val)
        label = self.text_encoder.int_to_char(label)
        return ''.join(label)

    def greedy_decode(self, x):
        x = np.argmax(x, axis=2)
        x = np.squeeze(x, axis=1)

        greedy_pred = []
        for i in range(len(x)-1):
            if x[i] == self.blank_val:
                continue
            elif x[i] != x[i+1] or i == len(x)-2:
                greedy_pred.append(x[i])

        greedy_pred = self.text_encoder.int_to_char(greedy_pred)
        return ''.join(greedy_pred)


















##################
