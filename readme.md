# ViNCE (Visual Interpretability for Neural Classifier Explainability) 

**ViNCE** is a neural network distillation and interpretability pipeline that compresses a deep CNN into a lightweight, interpretable decision tree, while retaining high accuracy and producing rich visual explanations of model behavior.

---

## 🧠 Research Question

Can we distill a high-performing convolutional neural network into a lightweight, interpretable decision tree while preserving predictive accuracy and generating actionable insight from model errors?

---

## 🚀 Features

- **Teacher model**: ResNet50 trained on the Imagenette dataset
- **Student model**: Decision Tree trained on PCA-reduced ResNet features
- **Distillation**: Teacher logits guide the student for fidelity and interpretability
- **VLM descriptor generation**:
  - Uses a vision-language model to name important PCA components
  - Logs prompts, raw responses, confidence/relevance-ranked candidates, and selected descriptors
  - Injects selected descriptors into decision tree feature labels
- **Explainability**:
  - Grad-CAM, Guided Backpropagation, and PEEK entropy maps
  - Composite image overlays with attention + uncertainty
- **Error analysis**:
  - Misclassification summary with confidence scores
  - Visual diagnostics for each misclassified sample
  - Automatic t-SNE + KMeans clustering of failure modes with optimal `k` search
- **Visualization tools**:
  - Confusion matrix, t-SNE plots, and decision tree diagrams
  - Auto-saved composite image grids (no need to close plots manually)

---

## 📂 Project Structure

```
project/
├── app.py                       # Main entry point
├── config.py                   # Hyperparameters & global config
├── data/
│   ├── prepare_data.py         # Dataset download, split, loaders
│   └── transforms.py           # Custom augmentations (e.g. Cutout)
├── models/
│   ├── resnet_teacher.py       # Transfer learning model
│   ├── decision_tree_student.py  # PCA + GridSearchCV
│   └── feature_extraction.py   # Extracts penultimate layer features
├── analyze/
│   ├── misclassifications.py   # Error visualization & summary
│   └── cluster_errors.py       # t-SNE + KMeans clustering of misclassifications
├── interpret/
│   ├── gradcam.py
│   ├── guided_backprop.py
│   ├── peek.py
│   └── composite.py            # Composite interpretability grid
├── visualize/
│   ├── confusion.py
│   ├── decision_tree_plot.py
│   └── tsne.py
└── utils/
    └── helpers.py              # Timing, denormalization, etc.
```

---

## ▶️ Usage

Run the full pipeline:

```bash
python app.py
```

It will:
- Train the teacher
- Distill and evaluate the student
- Generate VLM-backed semantic descriptors for tree-relevant PCA components (if `OPENAI_API_KEY` is set)
- Extract misclassifications
- Auto-generate Grad-CAM, Guided BP, PEEK visualizations
- Cluster errors via t-SNE
- Save everything to the `outputs/` directory

### Optional environment variables for VLM

```bash
export OPENAI_API_KEY=...
export VINCE_VLM_ENABLED=1
export VINCE_VLM_MODEL=gpt-4.1-mini
export VINCE_VLM_COMPONENTS_TO_DESCRIBE=20
export VINCE_VLM_TOP_SAMPLES_PER_COMPONENT=4
export VINCE_VLM_DESCRIPTOR_CONFIDENCE_THRESHOLD=0.55
```

---

## 📊 Outputs

You’ll get:
- `outputs/decision_tree.svg` — student model visualization
- `outputs/vlm_descriptors/descriptor_summary.json` — per-component prompts, raw VLM responses, candidates, and selected descriptors
- `outputs/vlm_descriptors/descriptor_analysis.json` — aggregate descriptor coverage/confidence analysis
- `outputs/misclassified_viz/` — Grad-CAM + PEEK overlays
- `outputs/tsne_misclassified_clusters.png` — clustered t-SNE plot
- `outputs/misclassified_summary.json` — structured error log

---

## 📈 Future Work

- Rank errors by entropy and confidence
- Add counterfactual perturbation analysis
- Auto-generate HTML reports or dashboards
- Integrate early exit prediction using latent activations

---

## 📄 License

MIT

---

## 🤝 Acknowledgements

Built using:
- PyTorch
- torchvision
- Captum
- scikit-learn
- matplotlib
- t-SNE / KMeans
