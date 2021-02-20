import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle



def prefix_beam_search(data, k=50):
    alphabet = ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    alpha = 0.2
    Pb = [defaultdict(lambda: 0) for i in range(data.shape[0])]
    Pnb = [defaultdict(lambda: 0) for i in range(data.shape[0])]

    with open('bigram-LM.pkl', 'rb') as f:
        lm = pickle.load(f)

    LM = lambda c1, c2: lm[c1+c2]

    Pb[0][''] = 1.0
    Pnb[0][''] = 0.0
    A_prev = ['']
    A_next = set()

    for t in range(data.shape[0]):
        for l in A_prev:
            for idx, c in enumerate(alphabet):
                if c == '-':
                    Pb[t][l] += data[t][idx] * (Pb[t-1][l] + Pnb[t-1][l])
                    A_next.add(l)
                else:
                    l_plus = l + c

                    if len(l) < 3:
                        Pnb[t][l_plus] += data[t][idx] * (Pb[t-1][l] + Pnb[t-1][l])
                    elif c == l[-1]:
                        Pnb[t][l] += data[t][idx] * Pnb[t-1][l]
                        Pnb[t][l_plus] += (LM(l_plus[-1], l_plus[-2])**alpha) * data[t][idx] * Pb[t-1][l]
                        A_next.add(l)
                    else:
                        Pnb[t][l_plus] += (LM(l_plus[-1], l_plus[-2])**alpha) * data[t][idx] * (Pb[t-1][l] + Pnb[t-1][l])

                    if l_plus not in A_prev:
                        Pb[t][l_plus] += data[t][0] * (Pb[t-1][l_plus] + Pnb[t-1][l_plus])
                        Pnb[t][l_plus] += data[t][idx] * Pnb[t-1][l_plus]

                    A_next.add(l_plus)

        A_next = sorted(A_next, key=lambda l: Pb[t][l] + Pnb[t][l], reverse=True)
        A_prev = A_next[:k]
        A_next = set()
    return A_prev
