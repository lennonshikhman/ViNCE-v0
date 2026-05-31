from __future__ import annotations

import csv

from vince.config import default_config
from vince.pipeline import run_dataset_pipeline
from vince.utils import ensure_dir, write_json


def main() -> None:
    config = default_config()
    ensure_dir(config.datasets_dir)
    ensure_dir(config.artifacts_dir)
    rows = []
    for name in config.dataset_names():
        metrics = run_dataset_pipeline(name, config)
        row = {"dataset": name, **metrics}
        rows.append(row)
    json_path = config.artifacts_dir / "summary_metrics.json"
    csv_path = config.artifacts_dir / "summary_metrics.csv"
    write_json(json_path, rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nCombined summary")
    print("dataset teacher_acc student_acc fidelity leaves")
    for row in rows:
        print(f"{row['dataset']} {row['teacher_accuracy_against_ground_truth']:.3f} {row['student_accuracy_against_ground_truth']:.3f} {row['teacher_student_fidelity']:.3f} {row['num_leaves']}")
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
