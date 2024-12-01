import os
import sys
import argparse
import logging
import math
import numpy as np
import pandas as pd
from itertools import combinations


def calculate_precision_recall_f1(true_labels, predicted_labels):
    """
    Calculate precision, recall, and F1-score without using external libraries.

    :param true_labels: Ground truth (list or numpy array of 0s and 1s)
    :param predicted_labels: Predicted values (list or numpy array of 0s and 1s)
    :return: Precision, Recall, F1-score
    """
    # True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    # Precision
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # Recall
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # F1-score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def load_raw_data(fname):
    data = pd.read_csv(fname)
    data.columns = data.columns.str.strip()  # Strip whitespace from column names
    # Parse month from date and add it as a feature
    data["month"] = pd.to_datetime(data["date"]).dt.month
    # Include all columns except 'date' and 'label' as features
    features = data.drop(columns=["date", "label"])
    labels = data["label"].to_numpy()
    return features, labels


def calculate_mean_and_variance(features, labels, target_label):
    selected_features = features[labels == target_label]
    mean = np.mean(selected_features, axis=0)
    variance = np.var(selected_features, axis=0)
    return mean, variance


def gaussian_probability(x, mean, variance):
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    return (1 / np.sqrt(2 * np.pi * variance)) * exponent


def train_naive_bayes(features, labels):
    params = {}
    for label in np.unique(labels):
        mean, variance = calculate_mean_and_variance(features, labels, label)
        prior = np.sum(labels == label) / len(labels)
        params[label] = {"mean": mean, "variance": variance, "prior": prior}
    return params


def predict_naive_bayes(features, params):
    predictions = []
    for x in features:
        posteriors = {}
        for label, param in params.items():
            likelihood = np.prod(
                gaussian_probability(x, param["mean"], param["variance"])
            )
            posterior = likelihood * param["prior"]
            posteriors[label] = posterior
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)


def report_metrics(test_labels, predictions, feature_set_name):
    precision, recall, f1 = calculate_precision_recall_f1(test_labels, predictions)
    logging.info(f"Feature Set: {feature_set_name}")
    logging.info(
        f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}"
    )
    return precision, recall, f1


def run(train_file, test_file):
    # Load training and testing data
    train_features, train_labels = load_raw_data(train_file)
    test_features, test_labels = load_raw_data(test_file)

    # Iterate over all combinations of features
    feature_names = train_features.columns
    best_f1 = 0
    best_features = None
    results = []  # To store results for the table

    for r in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, r):
            selected_train_features = train_features[list(combo)].to_numpy()
            selected_test_features = test_features[list(combo)].to_numpy()

            # Train and predict
            params = train_naive_bayes(selected_train_features, train_labels)
            predictions = predict_naive_bayes(selected_test_features, params)
            _, _, f1 = report_metrics(test_labels, predictions, f"Features: {combo}")

            # Append results for table
            results.append({"Feature Combination": combo, "F1-Score": f1})

            # Check if this is the best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_features = combo

    # Print the best feature combination and its F1 score
    print(f"Best F1 Score: {best_f1:.2f} with features: {best_features}")

    # Convert results to a DataFrame and print as a table
    results_df = pd.DataFrame(results)
    print("\nF1-Score Table:")
    print(results_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))



def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--training",
        required=True,
        metavar="<file path to the training dataset>",
        help="File path of the training dataset",
    )
    parser.add_argument(
        "-u",
        "--testing",
        required=True,
        metavar="<file path to the testing dataset>",
        help="File path of the testing dataset",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
        type=str,
        default="INFO",
    )
    args = parser.parse_args()
    return args


def main():
    args = command_line_args()
    logging.basicConfig(
        level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists(args.training):
        logging.error(f"The training dataset does not exist: {args.training}")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error(f"The testing dataset does not exist: {args.testing}")
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()

