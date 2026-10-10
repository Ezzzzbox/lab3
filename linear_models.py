# ф-ия перемешивает данные на тренировочные, тестовые и тд
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

# собственно линейная регрессия
from sklearn.linear_model import LinearRegression

# оценка метрики
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    model = LinearRegression()
    model.fit(data.train_features, data.train_target)
    target_predict = model.predict(data.test_features)
    dat


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    raise NotImplementedError


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    raise NotImplementedError
