import os
import sys
import argparse
import logging
import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def load_raw_data(fname):
    data = pd.read_csv(fname)
    data.columns = data.columns.str.strip()  # Strip whitespace from column names
    features = data[['avg (temperature)', 'power']].to_numpy()
    labels = data['label'].to_numpy()
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
            likelihood = np.prod(gaussian_probability(x, param["mean"], param["variance"]))
            posterior = likelihood * param["prior"]
            posteriors[label] = posterior
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)


def report_metrics(test_labels, predictions, feature_set_name):
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    logging.info(f"Feature Set: {feature_set_name}")
    logging.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    return precision, recall, f1


def run(train_file, test_file):
    # Load training and testing data
    train_features, train_labels = load_raw_data(train_file)
    test_features, test_labels = load_raw_data(test_file)

    # Feature Set 1: avg (temperature) only
    train_features_1 = train_features[:, [0]]  # Only temperature
    test_features_1 = test_features[:, [0]]
    params_1 = train_naive_bayes(train_features_1, train_labels)
    predictions_1 = predict_naive_bayes(test_features_1, params_1)
    precision_1, recall_1, f1_1 = report_metrics(test_labels, predictions_1, "Feature Set 1")

    # Feature Set 2: avg (temperature) and power
    params_2 = train_naive_bayes(train_features, train_labels)
    predictions_2 = predict_naive_bayes(test_features, params_2)
    precision_2, recall_2, f1_2 = report_metrics(test_labels, predictions_2, "Feature Set 2")

    # Summarize results
    results = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Feature Set 1": [precision_1, recall_1, f1_1],
        "Feature Set 2": [precision_2, recall_2, f1_2]
    })
    print(results)


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
    logging.basicConfig(level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(args.training):
        logging.error(f"The training dataset does not exist: {args.training}")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error(f"The testing dataset does not exist: {args.testing}")
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()