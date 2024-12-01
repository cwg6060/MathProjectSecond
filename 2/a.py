import os
import sys
import argparse
import logging


# 체감온도 계산 함수 (습도, 온도를 사용)
def calculate_apparent_temperature(temperature, humidity):
    return 13.12 + 0.6215 * temperature - 0.3965 * temperature * humidity


# 전날 대비 전력량 증가 피처 추가
def add_power_increase_feature(instances, threshold=10):
    # 전날 대비 전력량 증가 여부 피처 추가
    prev_power = None
    power_increase_feature = []
    for instance in instances:
        current_power = instance[7]  # 전력량은 8번째 칼럼(0-based index)

        if prev_power is not None:
            power_increase = current_power - prev_power
            # 전날 대비 전력량 증가가 threshold 이상이면 1, 아니면 0
            power_increase_feature.append(1 if power_increase >= threshold else 0)
        else:
            # 첫 번째 인스턴스는 비교할 전날 데이터가 없으므로 0으로 처리
            power_increase_feature.append(0)

        prev_power = current_power

    # 전력량 증가 피처를 인스턴스에 추가
    for i, instance in enumerate(instances):
        instance.append(power_increase_feature[i])

    return instances


def training(instances, labels):
    # 모델 학습 함수 구현
    # 현재는 기본 구현만 되어있음
    pass


def predict(instance, parameters):
    # 예측 함수 구현
    pass


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

    # 피처 엔지니어링 추가
    instances = add_power_increase_feature(instances)
    for instance in instances[:5]:  # 처음 5개 샘플을 출력해서 확인
        logging.debug(f"Feature-engineered instance: {instance}")

    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)

    # 피처 엔지니어링 적용 (테스트 데이터도 전날 대비 전력량 증가 피처 추가)
    instances = add_power_increase_feature(instances)

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
