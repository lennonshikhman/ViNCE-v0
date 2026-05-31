from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from .utils import ensure_dir


def fit_transform_pca(train_features: np.ndarray, test_features: np.ndarray, n_components: int, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    n = min(n_components, train_features.shape[0], train_features.shape[1])
    pca = PCA(n_components=n, random_state=0)
    train_z = pca.fit_transform(train_features)
    test_z = pca.transform(test_features)
    joblib.dump(pca, out_dir / "pca.joblib")
    np.savez_compressed(out_dir / "compressed_train.npz", X=train_z)
    np.savez_compressed(out_dir / "compressed_test.npz", X=test_z)
    np.save(out_dir / "explained_variance_ratio.npy", pca.explained_variance_ratio_)
    return {"pca": pca, "train": train_z, "test": test_z, "explained_variance_ratio": pca.explained_variance_ratio_}
