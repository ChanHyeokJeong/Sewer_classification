import os
import argparse
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_train = np.load(os.path.join(args.feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.feature_dir, "y_train.npy"))

    X_val = np.load(os.path.join(args.feature_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.feature_dir, "y_val.npy"))

    X_test = np.load(os.path.join(args.feature_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.feature_dir, "y_test.npy"))

    class_path = os.path.join(args.feature_dir, "class_names.txt")

    with open(class_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    print("Feature directory =", args.feature_dir)
    print("Output directory =", args.output_dir)
    print("Classes =", class_names)
    print("X_train shape =", X_train.shape)
    print("X_val shape =", X_val.shape)
    print("X_test shape =", X_test.shape)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )

    clf.fit(X_train_scaled, y_train)

    y_val_pred = clf.predict(X_val_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_precision = precision_score(y_test, y_test_pred, average="macro")
    test_recall = recall_score(y_test, y_test_pred, average="macro")

    report = classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(y_test, y_test_pred)

    print()
    print("DINOv3 Classification Result")
    print("Validation Accuracy =", round(val_acc, 4))
    print("Validation Macro-F1 =", round(val_f1, 4))

    print()
    print("Test Accuracy =", round(test_acc, 4))
    print("Test Macro-F1 =", round(test_f1, 4))
    print("Test Precision =", round(test_precision, 4))
    print("Test Recall =", round(test_recall, 4))

    print()
    print("Classification Report")
    print(report)

    print()
    print("Confusion Matrix")
    print(cm)

    result_path = os.path.join(args.output_dir, "result_dinov3_classifier.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("DINOv3 Classification Result\n")
        f.write(f"Feature directory = {args.feature_dir}\n")
        f.write(f"Classes = {class_names}\n")
        f.write(f"X_train shape = {X_train.shape}\n")
        f.write(f"X_val shape = {X_val.shape}\n")
        f.write(f"X_test shape = {X_test.shape}\n\n")

        f.write(f"Validation Accuracy = {val_acc:.4f}\n")
        f.write(f"Validation Macro-F1 = {val_f1:.4f}\n\n")

        f.write(f"Test Accuracy = {test_acc:.4f}\n")
        f.write(f"Test Macro-F1 = {test_f1:.4f}\n")
        f.write(f"Test Precision = {test_precision:.4f}\n")
        f.write(f"Test Recall = {test_recall:.4f}\n\n")

        f.write("Classification Report\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))

    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)

    print()
    print("Saved result file =", result_path)


if __name__ == "__main__":
    main()
