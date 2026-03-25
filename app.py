# app.py

import torch
from data.prepare_data import create_dataloaders
from models.resnet_teacher import build_resnet50_teacher, train_teacher
from models.feature_extraction import get_feature_extractor, extract_features
from models.decision_tree_student import apply_pca, train_decision_tree, evaluate_student
from visualize.decision_tree_plot import plot_decision_tree, get_used_features
from visualize.confusion import plot_confusion_matrix, compute_kappa
from visualize.tsne import plot_tsne
from utils.helpers import format_elapsed_time
from config import device, IMAGENETTE_CLASSES, TREE_VISUALIZATION_PATH
from analyze.misclassifications import get_misclassified_indices, save_misclassified_summary, visualize_misclassifications
from analyze.cluster_errors import tsne_cluster_misclassifications
from interpret.gradcam import compute_gradcam
from interpret.guided_backprop import compute_guided_backprop
from interpret.peek import compute_peek_map, compute_peek_overlay
from interpret.composite import plot_composite_grid
from analyze.vlm_descriptors import (
    analyze_descriptor_quality,
    build_descriptor_feature_names,
    generate_vlm_component_descriptors,
)
import time
import sys
import os
from config import VLM_ENABLED

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    total_start = time.time()

    # 1. Load Data
    print("\n>>> Loading and preparing data...")
    train_loader, val_loader, test_loader, dataset_test = create_dataloaders()

    # 2. Build & Train Teacher
    print("\n>>> Building and training teacher model...")
    teacher = build_resnet50_teacher()
    teacher = train_teacher(teacher, train_loader, val_loader, device)

    # 3. Feature Extraction
    print("\n>>> Extracting features...")
    extractor = get_feature_extractor(teacher)
    feats_train, logits_train = extract_features(extractor, train_loader, teacher, device)
    feats_val, logits_val = extract_features(extractor, val_loader, teacher, device)
    feats_test, logits_test = extract_features(extractor, test_loader, teacher, device)

    # 4. Train Student Model
    print("\n>>> Training decision tree student model...")
    feats_train_pca, feats_val_pca, feats_test_pca, pca_model = apply_pca(feats_train, feats_val, feats_test)
    teacher_preds_train = logits_train.argmax(axis=1)
    teacher_preds_test = logits_test.argmax(axis=1)

    student = train_decision_tree(feats_train_pca, teacher_preds_train)

    # 5. Evaluation
    print("\n>>> Evaluating student model...")
    y_true_test = [label for _, label in dataset_test]
    y_pred_test, student_acc, fidelity, teacher_acc = evaluate_student(
        student, feats_test_pca, y_true_test, teacher_preds_test
    )


    print("\n>>> Confusion Matrix and Kappa Score")
    plot_confusion_matrix(y_true_test, y_pred_test, class_names=IMAGENETTE_CLASSES)
    kappa = compute_kappa(y_true_test, y_pred_test)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # 6. Optional VLM descriptors for PCA components
    descriptor_records = []
    if VLM_ENABLED:
        print("\n>>> Generating VLM semantic descriptors for PCA components...")
        descriptor_records = generate_vlm_component_descriptors(
            train_pca_features=feats_train_pca,
            pca_model=pca_model,
            train_dataset=train_loader.dataset,
            used_feature_indices=get_used_features(student),
        )
        analyze_descriptor_quality(descriptor_records)

    # 7. Visualize Tree
    print("\n>>> Visualizing decision tree...")
    if descriptor_records:
        feature_names = build_descriptor_feature_names(student.n_features_in_, descriptor_records)
    else:
        feature_names = [f"PC {i}" for i in range(student.n_features_in_)]
    plot_decision_tree(student, feature_names, IMAGENETTE_CLASSES, save_path=str(TREE_VISUALIZATION_PATH))

    # 8. t-SNE
    print("\n>>> Plotting t-SNE projection of features...")
    plot_tsne(feats_test_pca, y_true_test, class_names=IMAGENETTE_CLASSES)

    # 9. Misclassification Analysis
    print("\n>>> Analyzing misclassifications...")
    misclassified_indices = get_misclassified_indices(teacher_preds_test, y_true_test)
    save_misclassified_summary(misclassified_indices, y_true_test, teacher_preds_test, logits_test, "outputs/misclassified_summary.json")

    # Required layers for interpretability
    pre_layer4 = torch.nn.Sequential(*list(teacher.children())[:7])
    spatial_extractor = teacher.layer4

    visualize_misclassifications(
        model=teacher,
        dataset=dataset_test,
        misclassified_indices=misclassified_indices,
        logits=logits_test,
        output_dir="outputs/misclassified_viz",
        class_names=IMAGENETTE_CLASSES,
        extractor=extractor,
        spatial_extractor=spatial_extractor,
        pre_layer4=pre_layer4,
        device=device
    )

    # 10. Misclassification Clustering
    print("\n>>> Clustering misclassifications with t-SNE...")
    tsne_cluster_misclassifications(feats_test, misclassified_indices, y_true_test)

    print("\n✅ Pipeline complete. Total time:", format_elapsed_time(time.time() - total_start))


if __name__ == '__main__':
    main()
