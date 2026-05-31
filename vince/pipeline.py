from __future__ import annotations

import numpy as np

from .compression import fit_transform_pca
from .config import ViNCEConfig
from .data import load_dataset, prepare_dataset
from .evaluation import compute_metrics
from .features import extract_or_load_features, save_feature_cache
from .student import RuleStudent
from .teacher import DatasetTeacherHead, TeacherModel
from .utils import ensure_dir, set_seed, write_json


def _refresh_dataset_teacher_predictions(caches: dict[str, dict], config: ViNCEConfig, artifact_dir) -> None:
    """Fit a dataset-label head on frozen train features and update teacher cache fields."""
    train_cache = caches["train"]
    teacher_head = DatasetTeacherHead(
        seed=config.seed,
        max_iter=config.teacher_head_max_iter,
        solver=config.teacher_head_solver,
        c=config.teacher_head_c,
    ).fit(train_cache["features"], train_cache["labels"])
    teacher_head.save(artifact_dir / "teacher")
    teacher_classes = teacher_head.model[-1].classes_
    for split, cache in caches.items():
        cache["teacher_classes"] = teacher_classes
        cache["teacher_logits"] = teacher_head.predict_logits(cache["features"])
        cache["teacher_preds"] = teacher_head.predict_labels(cache["features"])
        save_feature_cache(artifact_dir / "features", split, cache)


def _extract_split_features(split: str, dataset, teacher: TeacherModel, artifact_dir, feature_cfg: dict, config: ViNCEConfig, class_names: list[str]) -> dict:
    return extract_or_load_features(
        dataset,
        split,
        teacher,
        artifact_dir / "features",
        feature_cfg,
        config.batch_size,
        config.num_workers,
        class_names,
    )


def run_dataset_pipeline(dataset_name: str, config: ViNCEConfig) -> dict:
    if dataset_name not in config.dataset_configs:
        raise ValueError(f"No dataset config for {dataset_name}")
    set_seed(config.seed)
    ds_cfg = config.dataset_configs[dataset_name]
    artifact_dir = ensure_dir(config.artifacts_dir / dataset_name)
    for sub in ["features", "compression", "student", "teacher", "descriptors", "results"]:
        ensure_dir(artifact_dir / sub)
    write_json(artifact_dir / "config_used.json", config.to_dict())

    prepare_dataset(dataset_name, config.datasets_dir, seed=config.seed)
    teacher = TeacherModel(device=config.device)
    bundle = load_dataset(
        dataset_name,
        config.datasets_dir,
        teacher.transform,
        max_train=ds_cfg.max_train_samples,
        max_test=ds_cfg.max_test_samples,
        max_val=ds_cfg.max_val_samples,
        seed=config.seed,
    )
    feature_cfg = {
        "cache_version": 4,
        "dataset": dataset_name,
        "teacher": config.teacher_model,
        "teacher_head_max_iter": config.teacher_head_max_iter,
        "teacher_head_solver": config.teacher_head_solver,
        "teacher_head_c": config.teacher_head_c,
        "seed": config.seed,
        "max_train_samples": ds_cfg.max_train_samples,
        "max_val_samples": ds_cfg.max_val_samples,
        "max_test_samples": ds_cfg.max_test_samples,
    }
    caches = {
        "train": _extract_split_features("train", bundle.train, teacher, artifact_dir, feature_cfg, config, bundle.class_names),
        "test": _extract_split_features("test", bundle.test, teacher, artifact_dir, feature_cfg, config, bundle.class_names),
    }
    if bundle.val is not None:
        caches["val"] = _extract_split_features("val", bundle.val, teacher, artifact_dir, feature_cfg, config, bundle.class_names)
    _refresh_dataset_teacher_predictions(caches, config, artifact_dir)
    train_cache = caches["train"]
    test_cache = caches["test"]

    compressed = fit_transform_pca(train_cache["features"], test_cache["features"], config.pca_components, artifact_dir / "compression")
    X_train, X_test = compressed["train"], compressed["test"]
    if config.target_mode == "teacher_hard":
        y_target = train_cache["teacher_preds"]
    elif config.target_mode == "ground_truth":
        y_target = train_cache["labels"]
    else:
        raise ValueError(f"Unsupported target_mode: {config.target_mode}")

    feature_names = [f"pc_{i}" for i in range(X_train.shape[1])]
    student = RuleStudent(config.tree_max_depth, config.tree_min_samples_leaf, random_state=config.seed).fit(X_train, y_target)
    student.save(artifact_dir / "student", feature_names)
    student_preds = student.predict(X_test)
    metrics = compute_metrics(
        y_true=test_cache["labels"],
        teacher_preds=test_cache["teacher_preds"],
        student_preds=student_preds,
        tree_model=student.model,
        X_test=X_test,
        num_classes=len(np.unique(np.concatenate([train_cache["labels"], test_cache["labels"]]))),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    metrics["num_val_samples_used"] = int(len(bundle.val)) if bundle.val is not None else 0
    write_json(artifact_dir / "results" / "metrics.json", metrics)
    # Keep a root-level copy because the artifact checklist names metrics.json at
    # dataset scope while the detailed metrics requirement names results/metrics.json.
    write_json(artifact_dir / "metrics.json", metrics)
    print(
        f"{dataset_name}: teacher_acc={metrics['teacher_accuracy_against_ground_truth']:.3f} "
        f"student_acc={metrics['student_accuracy_against_ground_truth']:.3f} "
        f"fidelity={metrics['teacher_student_fidelity']:.3f} leaves={metrics['num_leaves']}"
    )

    if config.generate_descriptors:
        from .descriptors import generate_descriptors

        generate_descriptors(dataset_name, bundle.train, X_train, student.model, artifact_dir / "descriptors", config)
    return metrics
