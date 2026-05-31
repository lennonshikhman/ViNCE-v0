# ViNCE MVP

ViNCE means **Visual Interpretability for Neural Classifier Explainability**. This repository is a small, auditable MVP for turning frozen image-classifier behavior into a compact rule-based student model and optional post-hoc visual descriptor labels.

> **Important interpretability caveat:** Descriptor-labeled rules are post-hoc semantic hypotheses. They may improve auditability, but they are not automatically faithful causal explanations of the teacher model.

## What this MVP implements

The default pipeline:

1. downloads or prepares an image dataset under `./datasets/`;
2. runs a frozen torchvision ResNet18 ImageNet backbone;
3. trains a lightweight dataset-label linear probe on frozen teacher features so teacher accuracy/fidelity are computed in the correct dataset label space;
4. extracts and caches penultimate teacher features, raw ImageNet logits, dataset teacher logits, and labels;
5. compresses features with PCA;
6. trains a shallow `sklearn.tree.DecisionTreeClassifier` student;
7. evaluates teacher accuracy, student accuracy, and teacher-student fidelity;
8. exports tree rules and JSON tree nodes;
9. optionally asks an OpenAI vision-capable model to label selected tree splits with short semantic hypotheses.

The ResNet18 backbone is not fine-tuned. A small scikit-learn linear probe is fitted on frozen features for each dataset; this fixes the label-space mismatch that would otherwise make ImageNet class IDs incomparable to CIFAR-10, Pets, Flowers102, Food-101, and Cats vs Dogs labels.

## Supported datasets

All datasets are stored in the top-level `./datasets/` folder next to `run_all_experiments.py`; no datasets are placed inside `vince/`.

- CIFAR-10 via `torchvision.datasets.CIFAR10(root=Path("datasets") / "cifar10", download=True)`.
- Cats vs Dogs via the official Microsoft ZIP URL, cleaned locally into an `ImageFolder` layout.
- Oxford-IIIT Pet via `torchvision.datasets.OxfordIIITPet(root=Path("datasets") / "oxford_iiit_pet", download=True)`.
- Flowers102 via `torchvision.datasets.Flowers102(root=Path("datasets") / "flowers102", download=True)`, with train, validation, and test splits loaded separately.
- Food-101 via `torchvision.datasets.Food101(root=Path("datasets") / "food101", download=True)`.

No Kaggle login, Kaggle API key, Hugging Face account, Google Drive link, manual browser download, or external account is required.

### Cats vs Dogs cleaning

Cats vs Dogs is downloaded from:

`https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip`

The ZIP is stored at `datasets/cats_vs_dogs/raw/kagglecatsanddogs_5340.zip`, extracted to `datasets/cats_vs_dogs/extracted/`, and cleaned into:

```text
datasets/cats_vs_dogs/cleaned/
  train/cat/
  train/dog/
  test/cat/
  test/dog/
```

Cleaning verifies each image with PIL, skips corrupted files, converts usable images to RGB, does not resize during cleaning, and writes `manifest.json` with kept/skipped files. The split is deterministic with a fixed seed and defaults to an 80/20 train/test split.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API key file

Descriptor generation is disabled by default, so `api.txt` is not required for the default run.

If you enable descriptors, create a local `api.txt` at the project root containing only the key:

```bash
cp api.txt.example api.txt
# edit api.txt so it contains only your OpenAI API key
```

`api.txt` is ignored by git and is never printed or logged. The reader raises a clear error if the file is missing or empty.

## Run all experiments

```bash
python run_all_experiments.py
```

No flags are required. Defaults are defined in `vince/config.py`. The default subsets are small enough to test the full system on CPU:

- CIFAR-10: 2,000 train / 500 test
- Cats vs Dogs: 2,000 train / 500 test
- Oxford-IIIT Pet: 2,000 train / 500 test
- Flowers102: 2,000 train / 500 validation / 500 test
- Food-101: 3,000 train / 750 test

The device default is `auto`, which uses CUDA when available and CPU otherwise.

## Descriptor generation

Open `vince/config.py` and set:

```python
generate_descriptors = True
```

The code uses the official OpenAI Python SDK Responses API with the configurable default model `gpt-4.1-mini`. For each selected tree split it sends only a few representative images per side and requests strict JSON:

```json
{
  "descriptor": "short human-readable visual concept",
  "left_description": "what images on the left side tend to show",
  "right_description": "what images on the right side tend to show",
  "confidence": 0.0,
  "failure_mode": "empty string if none, otherwise why the descriptor may be unreliable"
}
```

Descriptors below the configured confidence threshold are stored but marked as abstained. Descriptor generation is a post-hoc labeling aid, not a causal or guaranteed-faithful explanation.

## Artifacts

Each run creates:

```text
artifacts/
  cifar10/
  cats_vs_dogs/
  oxford_iiit_pet/
  flowers102/
  food101/
  summary_metrics.csv
  summary_metrics.json
```

Each dataset folder contains:

```text
config_used.json
features/
compression/
student/
teacher/
descriptors/
results/metrics.json
```

Feature caches are `.npz` files containing frozen-backbone features, labels, raw ImageNet logits/predictions, dataset-label teacher logits/predictions, teacher class order, sample indices, class names, and a config hash; Flowers102 also writes a `val_features.npz` cache. The `teacher/` folder stores the dataset-label linear probe plus diagnostics such as solver settings, iteration counts, and convergence warnings. PCA artifacts include the model, compressed arrays, and explained variance ratio. Student artifacts include a fitted tree, text rules, and JSON nodes/root-to-leaf paths.

## Metrics

`metrics.json` includes:

- `teacher_accuracy_against_ground_truth`: dataset-label teacher-head accuracy against dataset labels. The head is trained on frozen ResNet18 features so predictions are in the same label space as each dataset.
- `student_accuracy_against_ground_truth`: decision-tree student accuracy against dataset labels.
- `teacher_student_fidelity`: agreement between student predictions and teacher hard labels.
- `fidelity_loss`: `1 - teacher_student_fidelity`.
- `macro_f1_student` / `macro_f1_teacher`: macro-F1 against ground truth.
- tree size metrics: depth, leaves, nodes, and mean path length.
- number of classes and train/test/validation samples used.

## Inspect rules

After running experiments:

```bash
python inspect_rules.py
```

This prints dataset names, tree rules, optional descriptor labels/confidence/abstention status, and root-to-leaf paths.
