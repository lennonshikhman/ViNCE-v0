from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, export_text

from .utils import ensure_dir, write_json


class RuleStudent:
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 20, random_state: int = 123) -> None:
        self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_target: np.ndarray) -> "RuleStudent":
        self.model.fit(X_train, y_target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def export_rules(self, feature_names: list[str]) -> str:
        return export_text(self.model, feature_names=feature_names)

    def to_json(self, feature_names: list[str]) -> dict[str, Any]:
        tree = self.model.tree_
        nodes = []
        for i in range(tree.node_count):
            feature = int(tree.feature[i])
            nodes.append({
                "node_id": i,
                "left": int(tree.children_left[i]),
                "right": int(tree.children_right[i]),
                "feature": None if feature == _tree.TREE_UNDEFINED else feature_names[feature],
                "threshold": None if feature == _tree.TREE_UNDEFINED else float(tree.threshold[i]),
                "samples": int(tree.n_node_samples[i]),
                "value": tree.value[i].tolist(),
                "is_leaf": bool(tree.children_left[i] == _tree.TREE_LEAF),
            })
        return {"nodes": nodes, "classes": self.model.classes_.tolist()}

    def root_to_leaf_paths(self, feature_names: list[str]) -> list[list[str]]:
        tree = self.model.tree_
        paths: list[list[str]] = []
        def walk(node: int, path: list[str]) -> None:
            if tree.children_left[node] == _tree.TREE_LEAF:
                paths.append(path + [f"leaf node {node}"])
                return
            name = feature_names[int(tree.feature[node])]
            thr = float(tree.threshold[node])
            walk(int(tree.children_left[node]), path + [f"node {node}: {name} <= {thr:.4f}"])
            walk(int(tree.children_right[node]), path + [f"node {node}: {name} > {thr:.4f}"])
        walk(0, [])
        return paths

    def save(self, out_dir: Path, feature_names: list[str]) -> dict:
        ensure_dir(out_dir)
        joblib.dump(self.model, out_dir / "decision_tree.joblib")
        rules = self.export_rules(feature_names)
        (out_dir / "rules.txt").write_text(rules, encoding="utf-8")
        tree_json = self.to_json(feature_names)
        tree_json["root_to_leaf_paths"] = self.root_to_leaf_paths(feature_names)
        write_json(out_dir / "tree.json", tree_json)
        return tree_json
