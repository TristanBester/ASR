def levenshtein_distance(a, b):
    '''Levenshtein distance is a metric that quantifies the diffence between two
    strings. *implementation case sensitive.

    Parameters:
    a (str): First string in comparison.
    b (str): Second string in comparison.

    Returns:
    int: Levenshtein distance.
    '''
    if len(a) == 0:
        return len(b)
    elif len(b) == 0:
        return len(a)
    elif a[0] == b[0]:
        return levenshtein_distance(a[1:], b[1:])
    else:
        return 1 + min(levenshtein_distance(a[1:], b),
                       levenshtein_distance(a, b[1:]),
                       levenshtein_distance(a[1:], b[1:]))


def word_error_count(ls_a, ls_b):
    '''Calulate the number of mismatched words - Levenshtein distance that considers
    word rather than character additions, deletions and substitutions.

    Parameters:
    ls_a (list): Words in first sentence.
    ls_b (list): Words in second sentence.

    Returns:
    int: Number of word errors.
    '''
    if not isinstance(ls_a, list):
        ls_a = ls_a.split(' ')
    if not isinstance(ls_b, list):
        ls_b = ls_b.split(' ')

    return levenshtein_distance(ls_a, ls_b)


def WER(a, b):
    '''Word error rate: (S + I + D)/N
    S - Number of word substitutions
    I - Number of word insertions
    D - Number of word deletions
    N - Number of words in second string.

    Parameters:
    a (str): First sentence.
    b (str): Second sentence.

    Returns:
    int: Word error rate.
    '''
    return word_error_count(a, b)/len(b.split(' '))


def CER(a, b):
    '''Character error rate: (S + I + D)/N
    S - Number of character substitutions
    I - Number of character insertions
    D - Number of character deletions
    N - Number of character in second string.

    Parameters:
    a (str): First sentence.
    b (str): Second sentence.

    Returns:
    int: Character error rate.
    '''
    return levenshtein_distance(a, b)/len(b)
