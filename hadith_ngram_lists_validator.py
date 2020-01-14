import json
import os
import numpy as np

import hadith_predictor as predictor


def validate_sanad_maten_lists(sanad_bigrams_path, sanad_unigrams_path,
                               maten_bigrams_path, maten_unigrams_path,
                               test_set_path, split_predictor,
                               min_tolerance=0, max_tolerance=5,
                               output_path=None, verbose=1):
    gold_split_positions = []
    predicted_split_positions = []

    test_set = np.load(test_set_path)

    if verbose:
        print("Testing on %d hadith" % (len(test_set)))

    hadith_number = 0
    for observation in test_set:
        hadith = observation[0]
        gold_split_position = observation[2]
        gold_split_positions.append(int(gold_split_position))
        predicted_split_position, _ = predictor.segment_hadith(hadith, sanad_bigrams_path, sanad_unigrams_path,
                                                               maten_bigrams_path, maten_unigrams_path,
                                                               split_predictor,
                                                               )

        predicted_split_positions.append(int(predicted_split_position))

        if verbose:
            if verbose == 2:
                print(str(hadith_number) + "_" + hadith + " predicted_split(%d)" % (int(predicted_split_position)))
            if verbose == 1:
                print("hadith_" + str(hadith_number) + " predicted_split(%d)" % (int(predicted_split_position)))
            hadith_number += 1

    accuracies = _get_accuracy_vary_tolerance(gold_split_positions, predicted_split_positions,
                                              min_tolerance=min_tolerance, max_tolerance=max_tolerance)

    info = {"Accuracies": accuracies,
            "Sanad_unigrams": sanad_unigrams_path,
            "Sanad_bigrams": sanad_bigrams_path,
            "Maten_unigrams": maten_unigrams_path,
            "Maten_bigrams": maten_bigrams_path,
            "Test_set": test_set_path,
            }

    if verbose == 2:
        print(info)

    if output_path:

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file_name = "lists_validation_output"
        output_file_number = 0
        final_path = output_path + output_file_name + "_" + str(output_file_number) + ".json"

        while os.path.exists(final_path):
            output_file_number += 1
            final_path = output_path + output_file_name + "_" + str(output_file_number) + ".json"

        fs = open(final_path, "w")
        json.dump(info, fs)

    return accuracies


def _get_accuracy_vary_tolerance(gold_splits, predicted_splits, min_tolerance=0, max_tolerance=5):
    accuracies = {}
    for tolerance in range(min_tolerance, max_tolerance + 1):
        accuracies[tolerance] = _split_position_accuracy(gold_splits, predicted_splits, tolerance)
    return accuracies


def _split_position_accuracy(gold_splits, predicted_splits, tolerance=3):
    correctly_segmented = []

    for (g, p) in zip(gold_splits, predicted_splits):
        if abs(g - p) <= tolerance:
            correctly_segmented.append(1)
        else:
            correctly_segmented.append(0)

    return len([res for res in correctly_segmented if res == 1]) / len(correctly_segmented) * 100
