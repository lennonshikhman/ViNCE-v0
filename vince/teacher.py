from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from .utils import ensure_dir, write_json


class TeacherModel:
    """Frozen ImageNet-pretrained backbone used by the ViNCE teacher."""

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.feature_model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device).eval()
        self.transform = self.weights.transforms()

    @torch.no_grad()
    def predict_logits(self, images: Tensor) -> Tensor:
        """Return raw ImageNet logits from the frozen pretrained classifier."""
        return self.model(images.to(self.device)).cpu()

    @torch.no_grad()
    def predict_labels(self, images: Tensor) -> Tensor:
        """Return raw ImageNet class IDs from the frozen pretrained classifier."""
        return self.predict_logits(images).argmax(dim=1)

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Tensor:
        feats = self.feature_model(images.to(self.device))
        return torch.flatten(feats, 1).cpu()


class DatasetTeacherHead:
    """Dataset-label teacher head trained on frozen ResNet18 features.

    The backbone remains frozen. This lightweight linear probe makes teacher labels
    live in the dataset label space, so teacher accuracy and student fidelity are
    meaningful for CIFAR-10, Pets, Flowers102, Food-101, and Cats vs Dogs.
    """

    def __init__(self, seed: int = 123, max_iter: int = 1000, solver: str = "lbfgs", c: float = 1.0) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.solver = solver
        self.c = c
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed, solver=solver, C=c),
        )
        self.diagnostics: dict = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DatasetTeacherHead":
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            raise ValueError(
                "DatasetTeacherHead requires at least two classes in the training subset; "
                f"got labels={unique_labels.tolist()}"
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(features, labels)
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        logistic = self.model[-1]
        self.diagnostics = {
            "solver": self.solver,
            "max_iter": self.max_iter,
            "C": self.c,
            "classes": logistic.classes_.tolist(),
            "n_iter": np.asarray(logistic.n_iter_).astype(int).tolist(),
            "converged": not convergence_warnings,
            "convergence_warnings": convergence_warnings,
            "num_train_samples": int(features.shape[0]),
            "num_features": int(features.shape[1]),
        }
        return self

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        scores = self.model.decision_function(features)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return np.asarray(scores)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def save(self, out_dir: Path) -> None:
        ensure_dir(out_dir)
        joblib.dump(self.model, out_dir / "dataset_teacher_head.joblib")
        write_json(out_dir / "dataset_teacher_head.json", self.diagnostics)
