import numpy as np

class TextEncoder():
    def __init__(self, blank_first=True):
        idx_shift = 96 if blank_first else 97
        self.char_int_mapping = {chr(i):(i-idx_shift) for i in range(97, 123)}
        self.char_int_mapping[' '] = 27 if blank_first else 26
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

    def decode_labels(self, labels, pad_val=0):
        decoded_labels = []
        for label in labels:
            label = self.__strip_padding(label, pad_val)
            decoded_labels.append(self.text_encoder.int_to_char(label))
        return self.collapse(decoded_labels)

    def greedy_decode(self, x):
        raw_preds = []
        for i in x:
            raw_pred = [np.argmax(t).item() for t in i]
            raw_preds.append(raw_pred + [-1])

        greedy_preds = []
        for pred in raw_preds:
            greedy_pred = []
            for i in range(len(pred)-1):
                if pred[i] == self.blank_val:
                    continue
                elif pred[i] != pred[i+1]:
                    greedy_pred.append(pred[i])
            greedy_preds.append(self.text_encoder.int_to_char(greedy_pred))
        return greedy_preds

    def collapse(self, ls):
        collapsed = [''] * len(ls)
        for i, pred in enumerate(ls):
            for ch in pred:
                collapsed[i] += ch
        return collapsed

















##################
