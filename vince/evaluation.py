from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def mean_path_length(tree_model, X: np.ndarray) -> float:
    indicator = tree_model.decision_path(X)
    return float(np.asarray(indicator.sum(axis=1)).ravel().mean())


def compute_metrics(y_true: np.ndarray, teacher_preds: np.ndarray, student_preds: np.ndarray, tree_model, X_test: np.ndarray, num_classes: int, n_train: int, n_test: int) -> dict:
    fidelity = accuracy_score(teacher_preds, student_preds)
    return {
        "teacher_accuracy_against_ground_truth": float(accuracy_score(y_true, teacher_preds)),
        "student_accuracy_against_ground_truth": float(accuracy_score(y_true, student_preds)),
        "teacher_student_fidelity": float(fidelity),
        "fidelity_loss": float(1.0 - fidelity),
        "macro_f1_student": float(f1_score(y_true, student_preds, average="macro", zero_division=0)),
        "macro_f1_teacher": float(f1_score(y_true, teacher_preds, average="macro", zero_division=0)),
        "tree_depth": int(tree_model.get_depth()),
        "num_leaves": int(tree_model.get_n_leaves()),
        "num_nodes": int(tree_model.tree_.node_count),
        "mean_path_length": mean_path_length(tree_model, X_test),
        "num_classes": int(num_classes),
        "num_train_samples_used": int(n_train),
        "num_test_samples_used": int(n_test),
    }
