from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def expand(array):
    ones = np.ones((array.shape[0], 1))
    return np.hstack([ones, array])


def Z_normalize(array):
    std = array.std(axis=0)
    std[std == 0] = 1
    return (array - array.mean(axis=0)) / std


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    np.random.seed(777)
    features_train = Z_normalize(expand(data.train_features.to_numpy()))
    targets_train = Z_normalize(data.train_target.to_numpy())
    features_test = Z_normalize(expand(data.test_features.to_numpy()))
    targets_test = Z_normalize(data.test_target.to_numpy())
    weights = np.ones((features_train.shape[1], 1))
    losses = []

    for _ in range(data.epochs):
        res = features_train @ weights
        grad = features_train.T @ (res - targets_train) / features_train.shape[0]
        weights -= data.learning_rate * grad

        res_test = features_test @ weights
        loss = (targets_test - res_test) ** 2
        losses.append(loss.mean())
    preds = features_test @ weights
    return RegressionResult(weights[1:], weights[0], losses, preds)


def minmax_normalize(array):
    return (array - array.min()) / (array.max() - array.min())


def logistic_probs(features, weights):
    mtrx = features @ weights
    return 1 / (1 + np.exp(-mtrx))


def logistic_res(probs):
    esp = 1e-5
    return np.clip(probs, esp, 1 - esp)


def logistic_loss(trgts, probs):
    return -(trgts * np.log(probs) + (1 - trgts) * np.log(1 - probs))


def ROC_AUC(trgts, probs):
    pos = probs[trgts == 1]
    neg = probs[trgts == 0]
    mtrx = pos[:, None] > neg[None, :]
    return mtrx.mean()


def metrics(trgts, res):
    tp = len([i for i in range(len(trgts)) if trgts[i] == res[i] and trgts[i] == 1])
    tn = len([i for i in range(len(trgts)) if trgts[i] == res[i] and trgts[i] == 0])
    fn = len([i for i in range(len(trgts)) if trgts[i] != res[i] and trgts[i] == 1])
    fp = len([i for i in range(len(trgts)) if trgts[i] != res[i] and trgts[i] == 0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return (accuracy, precision, recall, f1)


def logic_grad(features, targets, weights):
    probs = logistic_probs(features, weights)
    res = logistic_res(probs)
    loss = logistic_loss(targets, res)
    grad = features.T @ loss / features.shape[0]
    return grad


# я не хочу это рефакторить
def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    features_train = expand(Z_normalize(data.train_features.to_numpy()))
    features_test = expand(Z_normalize(data.test_features.to_numpy()))
    targets_train = data.train_target.to_numpy()
    targets_test = data.test_target.to_numpy()
    weights = np.ones((features_train.shape[1], 1))
    losses = []
    ros = []
    mm = []

    for _ in range(data.epochs):
        probs_train = logistic_probs(features_train, weights)
        res_train = logistic_res(probs_train)
        grad = (
            features_train.T @ (probs_train - targets_train) / features_train.shape[0]
        )
        weights = weights - data.learning_rate * grad
        ros.append(ROC_AUC(targets_train, probs_train))
        metric = metrics(targets_train, res_train)
        mm.append(metric)

        res_test = logistic_res(logistic_probs(features_test, weights))
        loss_test = logistic_loss(targets_test, res_test).mean()
        losses.append(loss_test)

    probabilities = logistic_probs(features_test, weights)
    results = logistic_res(probabilities)
    return ClassificationResult(
        weights[1:],
        weights[0],
        losses,
        probabilities,
        results,
    )


def _metrics(target, probabilities) -> ClassificationMetrics:
    """Пять метрик классификации по вероятностям и порогу 0.5."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    target = np.asarray(target).astype(int)
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions)),
        recall=float(recall_score(target, predictions)),
        f1=float(f1_score(target, predictions)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    features_train = np.array(data.train_features)
    features_test = np.array(data.test_features)
    targets_train = np.array(data.train_target)
    targets_test = np.array(data.test_target)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=0),
        "decision_tree": DecisionTreeClassifier(random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        model.fit(features_train, targets_train)
        results[name] = _metrics(targets_test, model.predict_proba(features_test)[:, 1])

    return BaselineComparisonResult(**results)


#
# def draw_linear_regression():
#     stds = pd.read_csv("Student_Performance.csv")
#     stds["Extracurricular Activities"] = stds["Extracurricular Activities"].map(
#         {"Yes": 1, "No": 0}
#     )
#     features = stds.drop(columns="Performance Index")
#     targets = stds[["Performance Index"]]
#     x_train, x_test, y_train, y_test = train_test_split(
#         features,
#         targets,
#         test_size=0.2,
#         train_size=0.8,
#     )
#     linear_input = RegressionInput(x_train, y_train, x_test, y_test)
#     linear_res = train_linear_regression(linear_input)
#     plt.plot(linear_res.loss_history)
#     plt.show()
#
#
# def draw_logistic_classifier():
#     df = pd.read_csv("german.csv", sep=";")
#     # breakpoint()
#     features = df.drop(columns="Creditability")
#     targets = df[["Creditability"]]
#     x_train, x_test, y_train, y_test = train_test_split(
#         features,
#         targets,
#         test_size=0.2,
#         train_size=0.8,
#     )
#     logistic_input = ClassificationInput(x_train, y_train, x_test, y_test)
#     logistic_res = train_logistic_classifier(logistic_input)
#     # breakpoint()
#     plt.plot(logistic_res.loss_history)
#     plt.show()


def print_metrics(metrics: ClassificationMetrics):
    print("roc_auc ", metrics.roc_auc)
    print("accuracy ", metrics.accuracy)
    print("precision ", metrics.precision)
    print("recall ", metrics.recall)
    print("f1 ", metrics.f1)


def draw_compare():
    df = pd.read_csv("german.csv", sep=";")
    # breakpoint()
    features = df.drop(columns="Creditability")
    targets = df[["Creditability"]]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        train_size=0.8,
    )
    logistic_input = ClassificationInput(x_train, y_train, x_test, y_test)
    my = train_logistic_classifier(logistic_input)
    my_metrics = _metrics(y_test, my.probabilities)
    bas = compare_classical_models(logistic_input)
    print("MY LOGIC")
    print_metrics(my_metrics)
    print()
    print("Tree")
    print_metrics(bas.decision_tree)
    print()
    print("kukulus")
    print_metrics(bas.knn)
    print()
    print("not mine")
    print_metrics(bas.logistic_regression)
    # plt.plot()
    # plt.show()


draw_compare()

# draw_linear_regression()
# draw_logistic_classifier()
