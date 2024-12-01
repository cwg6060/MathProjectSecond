import os
import sys
import argparse
import logging
import math


def training(instances, labels):
    # 클래스별 데이터 분리
    class_0_instances = []
    class_1_instances = []

    for i, label in enumerate(labels):
        if label == 0:
            class_0_instances.append([float(x) for x in instances[i]])
        else:
            class_1_instances.append([float(x) for x in instances[i]])

    # 평균과 분산 계산
    def calculate_mean_and_variance(instances):
        if not instances:
            return [], []
        mean = [sum(feature) / len(feature) for feature in zip(*instances)]
        variance = [
            max(sum((float(x) - m) ** 2 for x in feature) / len(feature), 1e-10)
            for feature, m in zip(zip(*instances), mean)
        ]
        return mean, variance

    mean_0, variance_0 = calculate_mean_and_variance(class_0_instances)
    mean_1, variance_1 = calculate_mean_and_variance(class_1_instances)

    # 클래스 사전확률
    prob_class_0 = len(class_0_instances) / len(instances)
    prob_class_1 = len(class_1_instances) / len(instances)

    parameters = {
        "mean_0": mean_0,
        "variance_0": variance_0,
        "prob_class_0": prob_class_0,
        "mean_1": mean_1,
        "variance_1": variance_1,
        "prob_class_1": prob_class_1,
    }

    return parameters


# 나이브 베이즈 예측 함수
def predict(instance, parameters):
    def gaussian_probability(x, mean, variance):
        if variance == 0:
            variance = 1e-10
        exponent = math.exp(-((float(x) - mean) ** 2) / (2 * variance))
        return (1 / math.sqrt(2 * math.pi * variance)) * exponent

    prob_0 = parameters["prob_class_0"]
    prob_1 = parameters["prob_class_1"]

    for i in range(len(instance)):
        prob_0 *= gaussian_probability(
            instance[i], parameters["mean_0"][i], parameters["variance_0"][i]
        )
        prob_1 *= gaussian_probability(
            instance[i], parameters["mean_1"][i], parameters["variance_1"][i]
        )

    return 0 if prob_0 > prob_1 else 1


def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))


def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels


def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)

    # report
    report(predictions, labels)


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--training",
        required=True,
        metavar="<file path to the training dataset>",
        help="File path of the training dataset",
        default="training.csv",
    )
    parser.add_argument(
        "-u",
        "--testing",
        required=True,
        metavar="<file path to the testing dataset>",
        help="File path of the testing dataset",
        default="testing.csv",
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
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()
