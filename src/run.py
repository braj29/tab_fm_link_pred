# run_experiment.py
import argparse
from data import prepare_data
from model import build_tabicl, build_tabpfn
from metrics import classification_accuracy, link_prediction_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tabicl",
        choices=["tabicl", "tabpfn"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=== Loading data (small experiment) ===")
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data()

    if args.model == "tabicl":
        print("=== Building TabICL ===")
        clf = build_tabicl()

    elif args.model == "tabpfn":
        print("=== Building TabPFN ===")
        clf = build_tabpfn(device=args.device)

    print("=== Fitting ===")
    clf.fit(X_train, y_train)

    print("=== Validation accuracy ===")
    val_acc = classification_accuracy(clf, X_valid, y_valid)
    print(f"Val Accuracy: {val_acc:.4f}")

    print("=== Test metrics ===")
    test_acc = classification_accuracy(clf, X_test, y_test)
    lp = link_prediction_metrics(clf, X_test, y_test)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(lp)


if __name__ == "__main__":
    main()
