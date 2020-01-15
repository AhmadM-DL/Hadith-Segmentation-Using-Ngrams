import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import json
import os
import sunnah_com_books_extractor as extractor

nltk.download('punkt')
nltk.download("stopwords")


def extract_sanad_maten_ngrams(books_paths, output_path, test_size_percent=0.25, top_frequent_percent=5, verbose=1,
                               remove_stop_words=0):
    books = {}

    if verbose:
        print("Reading Books JSON files")

    for book_path in books_paths:
        book_dictionary = json.load(open(book_path, "r"))
        sanad, maten, _ = extractor.book_maten_sanad_atraf_extractor(book_dictionary)
        sanad = sanad.split("\n")
        maten = maten.split("\n")

        # Take only hadith that contains sanad (as some doesn't)
        # get empty sanad indices
        indices_empty_sanad = [i for i, s in enumerate(sanad) if s == '']
        sanad = [s for i, s in enumerate(sanad) if not i in indices_empty_sanad]
        maten = [m for i, m in enumerate(maten) if not i in indices_empty_sanad]

        books[book_dictionary["Title"]] = {
            "sanad": sanad,
            "maten": maten,
        }

        if verbose:
            if book_dictionary["Title"]:
                print(book_dictionary["Title"])
            else:
                print("Read book with no title")

    if verbose:
        print("Extracting sanad/maten ngrams + test data")

    s_bi, s_uni, m_bi, m_uni, test_set = _extract_sanad_maten_ngrams(books, test_size_percent=test_size_percent,
                                                                     top_frequent_percent=top_frequent_percent,
                                                                     remove_stop_words=remove_stop_words)

    if verbose:
        print("Writing sanad/maten ngrams files")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(output_path + "/sanad_bigrams.npy", s_bi)
    np.save(output_path + "/sanad_unigrams.npy", s_uni)
    np.save(output_path + "/maten_bigrams.npy", m_bi)
    np.save(output_path + "/maten_unigrams.npy", m_uni)

    if verbose:
        print("Writing test data file")

    np.save(output_path + "/test_set.npy", test_set)

    if verbose:
        print("Writing extraction configuration file")

    json.dump({"books_used": books_paths,
               "top_frequent_percent": top_frequent_percent,
               "test_size_percent": test_size_percent},
              open(output_path + "/cfg.json", "w"))
    return s_bi, s_uni, m_bi, m_uni, test_set


def _extract_sanad_maten_ngrams(books_dictionary, test_size_percent=0.25, remove_stop_words=0,
                                top_frequent_percent=5, verbose=1):
    # Get training books preliminary data (sanad/maten)
    all_sanad = [book_content["sanad"] for (_, book_content) in books_dictionary.items()]
    all_maten = [book_content["maten"] for (_, book_content) in books_dictionary.items()]

    # Flatten Preliminary data
    preliminary_sanad = [sanad for sanad_list in all_sanad for sanad in sanad_list]
    preliminary_maten = [maten for maten_list in all_maten for maten in maten_list]

    if verbose:
        print("The book contained %d hadith with sanad" % (len(preliminary_sanad)))

    # Split Preliminary data into train_test
    sanad_train, sanad_test, maten_train, maten_test = train_test_split(preliminary_sanad,
                                                                        preliminary_maten,
                                                                        test_size=test_size_percent)
    if verbose:
        print("Train-Test Split (%f): %d train, %d test" % (test_size_percent, len(sanad_train), len(sanad_test)))

    # Get Final Sanad Bigram Precompiled Lists
    sanad_bigrams = _generate_ngrams_from_sets(sanad_train, ngrams_number=2,
                                               top_frequent_percent=top_frequent_percent)
    sanad_bigrams = [(a[0], a[1]) for a in sanad_bigrams]

    # TODO work on verbose printing

    # Get Final Sanad unigram Precompiled Lists
    sanad_unigrams = _generate_ngrams_from_sets(sanad_train, ngrams_number=1, remove_stop_words=remove_stop_words,
                                                top_frequent_percent=top_frequent_percent)

    # Get Final Maten Bigram Precompiled Lists
    maten_bigrams = _generate_ngrams_from_sets(maten_train, ngrams_number=2)
    maten_bigrams = [(a[0], a[1]) for a in maten_bigrams]

    # Get Final Maten unigram Precompiled Lists
    maten_unigrams = _generate_ngrams_from_sets(maten_train, ngrams_number=1)

    # Get test set
    test_set = _generate_test_set(sanad_test, maten_test)

    output = sanad_bigrams, sanad_unigrams, maten_bigrams, maten_unigrams, test_set
    return output


def _generate_test_set(sanad_set, maten_set):
    test_set = []
    for i in range(len(sanad_set)):
        hadith = sanad_set[i].strip() + " " + maten_set[i].strip()

        tokenized_sanad = word_tokenize(sanad_set[i])
        sanad_split_token = tokenized_sanad[-1]
        sanad_split_token_position = len(tokenized_sanad) - 1

        test_set.append((hadith, sanad_split_token, sanad_split_token_position))
    return test_set


def _generate_ngrams_from_sets(hadith_part_set, ngrams_number, top_frequent_percent=0, remove_stop_words=0):
    # Get grams from training set
    grams = [list(ngrams(word_tokenize(sanad), ngrams_number)) for sanad in hadith_part_set]

    # Flatten sanad_grams lists
    grams = [bigram for bigram_list in grams for bigram in bigram_list]

    # Get unique sanad_train_grams and thier counts
    unique_grams, unique_grams_counts = np.unique(grams, return_counts=True, axis=0)

    if remove_stop_words:
        # Get Arabic Stop Words
        arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
        if ngrams_number == 1:
            non_stop_words_indices = [i for i, s in enumerate(unique_grams) if s not in arb_stopwords]
        else:
            non_stop_words_indices = [i for i, s in enumerate(unique_grams) if
                                      _number_tuple_elements_in_list(s, arb_stopwords) < len(s)]

        unique_grams = unique_grams[non_stop_words_indices]
        unique_grams_counts = unique_grams_counts[non_stop_words_indices]

    # Sort unique_sanad_grams and get top &top_frequent_percent grams
    sorted_unique_grams = unique_grams[np.argsort(unique_grams_counts)[::-1]]

    if top_frequent_percent:
        grams_top_frequent_size = int((top_frequent_percent * len(unique_grams)) // 100)
        frequent_grams = sorted_unique_grams[:grams_top_frequent_size]
        return frequent_grams

    return unique_grams


def _number_tuple_elements_in_list(mtuple, mlist):
    number_tuple_elements_in_list = 0
    for element in mtuple:
        if element in mlist:
            number_tuple_elements_in_list += 1
    return number_tuple_elements_in_list
