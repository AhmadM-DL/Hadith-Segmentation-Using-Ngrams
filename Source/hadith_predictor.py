import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np
import pyarabic.araby as araby
import unicodedata as ud
from termcolor import colored
import math

nltk.download('punkt')

PREDICTOR_INFORMATION_GAIN = "information_gain"
PREDICTOR_BASELINE = "baseline"

SANAD_LABEL = "SANAD"
MATEN_LABEL = "MATEN"
UNCERTAIN_LABEL = "UNCERTAIN"


def segment_hadith(hadith, sanad_bigrams_npy_file_path,
                   sanad_unigrams_npy_file_path,
                   maten_bigrams_npy_file_path,
                   maten_unigrams_npy_file_path,
                   split_position_predictor=PREDICTOR_INFORMATION_GAIN,
                   verbose=0):
    sanad_bigrams = np.load(sanad_bigrams_npy_file_path)
    sanad_unigrams = np.load(sanad_unigrams_npy_file_path)
    maten_bigrams = np.load(maten_bigrams_npy_file_path)
    maten_unigrams = np.load(maten_unigrams_npy_file_path)

    split_position, split_word = _segment_hadith(hadith, sanad_bigrams,
                                                 sanad_unigrams,
                                                 maten_bigrams,
                                                 maten_unigrams,
                                                 split_position_predictor=split_position_predictor,
                                                 )
    if verbose:
        print_annotated_hadith(hadith, split_position)

    return split_position, split_word


def pre_process_hadith(hadith):
    # Remove White Space
    hadith = hadith.replace("\n", "").replace("\t", "")
    # Remove Tashkeel
    hadith = araby.strip_tashkeel(hadith)
    # Remove Punctuation
    hadith = ''.join(c for c in hadith if not ud.category(c).startswith('P'))
    # Tokenize Hadith
    hadith_tokens = word_tokenize(hadith)
    return hadith_tokens


def _segment_hadith(hadith, sanad_bigrams, sanad_unigrams, maten_bigrams, maten_unigrams,
                    split_position_predictor=PREDICTOR_BASELINE):
    # Prepare Hadith
    hadith_tokens = pre_process_hadith(hadith)

    # Get Hadith Bigrams
    hadith_bigrams = list(ngrams(hadith_tokens, 2))

    # Loop on each bigram and label it "sanad" or "maten" if it belongs to sanad or maten
    annotated_hadith = []
    for bigram in hadith_bigrams:
        if bigram in sanad_bigrams:
            word1 = (bigram[0], SANAD_LABEL)
            word2 = (bigram[1], SANAD_LABEL)
            annotated_hadith.append(word1)
            annotated_hadith.append(word2)
        elif bigram in maten_bigrams:
            word1 = (bigram[0], MATEN_LABEL)
            word2 = (bigram[1], MATEN_LABEL)
            annotated_hadith.append(word1)
            annotated_hadith.append(word2)
        else:
            unigrams = list(bigram)
            for unigram in unigrams:
                if unigram in sanad_unigrams:
                    unigram = (unigram, SANAD_LABEL)
                    annotated_hadith.append(unigram)
                elif unigram in maten_unigrams:
                    unigram = (unigram, MATEN_LABEL)
                    annotated_hadith.append(unigram)
                else:
                    unigram = (unigram, UNCERTAIN_LABEL)
                    annotated_hadith.append(unigram)

    annotated_hadith = _remove_duplicated_tokens(annotated_hadith)

    if split_position_predictor == PREDICTOR_BASELINE:
        split_position = _split_position_baseline_predictor(annotated_hadith)

    if split_position_predictor == PREDICTOR_INFORMATION_GAIN:
        split_position = _split_position_information_gain_predictor(annotated_hadith)

    return split_position, annotated_hadith[split_position][0]


def print_annotated_hadith(hadith, hadith_token_split_position):
    # Prepare Hadith
    hadith_tokens = pre_process_hadith(hadith)

    for i, token in enumerate(hadith_tokens):
        if i <= int(hadith_token_split_position):
            print(colored(token, 'red'), end=" ")
        else:
            print(colored(token, 'blue'), end=" ")


def _remove_duplicated_tokens(annotated_hadith_tokens):
    fixed_annotated_hadith_tokens = []
    i = 0
    while i < len(annotated_hadith_tokens):
        token = annotated_hadith_tokens[i][0]
        label = annotated_hadith_tokens[i][1]

        if i + 1 >= len(annotated_hadith_tokens):
            fixed_annotated_hadith_tokens.append((token, label))
            break

        next_token = annotated_hadith_tokens[i + 1][0]
        next_label = annotated_hadith_tokens[i + 1][1]
        if token == next_token:
            fixed_annotated_hadith_tokens.append((token, _annotation_resolver(label, next_label)))
            i += 1
        else:
            fixed_annotated_hadith_tokens.append((token, label))
        i += 1

    return fixed_annotated_hadith_tokens


def _annotation_resolver(annotation1, annotation2):
    if annotation1 == SANAD_LABEL and annotation2 == SANAD_LABEL: return SANAD_LABEL
    if annotation1 == MATEN_LABEL and annotation2 == MATEN_LABEL: return MATEN_LABEL
    if annotation1 == UNCERTAIN_LABEL and annotation2 == UNCERTAIN_LABEL: return UNCERTAIN_LABEL

    if annotation1 == SANAD_LABEL and annotation2 == UNCERTAIN_LABEL: return SANAD_LABEL
    if annotation1 == MATEN_LABEL and annotation2 == UNCERTAIN_LABEL: return MATEN_LABEL
    if annotation1 == UNCERTAIN_LABEL and annotation2 == SANAD_LABEL: return SANAD_LABEL
    if annotation1 == UNCERTAIN_LABEL and annotation2 == MATEN_LABEL: return MATEN_LABEL

    if annotation1 == SANAD_LABEL and annotation2 == MATEN_LABEL: return UNCERTAIN_LABEL
    if annotation1 == MATEN_LABEL and annotation2 == SANAD_LABEL: return UNCERTAIN_LABEL

    print("annotation_resolver: Error unkown annotations (%s,%s)" % (annotation1, annotation2))
    return UNCERTAIN_LABEL


def _split_position_baseline_predictor(labeled_hadith_tokens):
    for i, _ in enumerate(labeled_hadith_tokens):

        if i + 1 >= len(labeled_hadith_tokens):
            break

        if labeled_hadith_tokens[i][1] == "MATEN":
            if labeled_hadith_tokens[i + 1][1] == "MATEN" or labeled_hadith_tokens[i + 1][1] == "UNCERTAIN":
                break

        if labeled_hadith_tokens[i][1] == "UNCERTAIN":
            if labeled_hadith_tokens[i + 1][1] == "MATEN" or labeled_hadith_tokens[i + 1][1] == "UNCERTAIN":
                break

    return i


def _split_position_information_gain_predictor(labeled_hadith_tokens):
    labels = [label for _, label in labeled_hadith_tokens]

    best_split = 0
    max_information_gain = 0

    for split_candidate in range(3, len(labeled_hadith_tokens) - 1):

        cur_information_gain = _information_gain(labels, split_candidate)

        if cur_information_gain > max_information_gain:
            max_information_gain = cur_information_gain
            best_split = split_candidate

    return best_split


def _entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)

    return ent


def _information_gain(seq, split_pos):
    parent_entropy = _entropy(seq)

    child1 = seq[:split_pos]
    child2 = seq[split_pos:]

    child1_entropy = _entropy(child1)
    child2_entropy = _entropy(child2)

    parent_split_entropy = len(child1) / len(seq) * child1_entropy + len(child2) / len(seq) * child2_entropy

    return parent_entropy - parent_split_entropy
