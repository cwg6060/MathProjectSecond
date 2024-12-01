import os
import sys
import argparse
import logging
import math


# 전날 대비 전력량 증가 피처 추가
def add_power_increase_feature(instances):
    threshold = 50  # 전력량 증가가 50 이상이면 1, 아니면 0 (데이터 특성에 맞게 조정)
    prev_power = None
    power_increase = []
    for instance in instances:
        current_power = float(instance[7])  # power column
        if prev_power is None:
            power_increase.append(0)
        else:
            power_increase.append(1 if current_power - prev_power > threshold else 0)
        prev_power = current_power
    return power_increase


# 체감온도 계산 (온도와 습도를 기반으로 계산)
def add_feels_like_temperature(instances):
    feels_like_temp = []
    for instance in instances:
        temperature = float(instance[1])  # avg temperature
        humidity = float(instance[4])  # avg humidity
        # 체감온도 계산식 수정 (기상청 체감온도 공식 적용)
        feels_like = temperature + 0.04 * humidity - 4.25
        feels_like_temp.append(feels_like)
    return feels_like_temp


# 데이터 로딩 시 피처들 추가
def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()  # Skip header
        for line in f:
            tmp = line.strip().split(", ")
            # Convert all numeric values to float
            for i in range(1, len(tmp) - 1):
                tmp[i] = float(tmp[i])
            instances.append(tmp[1:-1])  # 날짜와 라벨 제외
            labels.append(int(tmp[-1]))  # label

    # 추가 피처
    power_increase = add_power_increase_feature(instances)
    feels_like_temp = add_feels_like_temperature(instances)

    # 피처 결합
    for i in range(len(instances)):
        instances[i].append(power_increase[i])
        instances[i].append(feels_like_temp[i])

    return instances, labels


# 가우시안 나이브 베이즈 모델 훈련
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
    precision = round(tp / (tp + fp), 2) * 100 if (tp + fp) > 0 else 0

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100 if (tp + fn) > 0 else 0

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))


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
