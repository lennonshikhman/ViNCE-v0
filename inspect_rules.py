from __future__ import annotations

from pathlib import Path

from vince.config import DATASET_NAMES
from vince.utils import read_json


def _descriptor_map(descriptor_dir: Path) -> dict[int, dict]:
    out = {}
    if not descriptor_dir.exists():
        return out
    for path in descriptor_dir.glob("node_*/node_descriptor.json"):
        data = read_json(path)
        out[int(data.get("node_id", path.parent.name.split("_")[-1]))] = data
    return out


def inspect_dataset(name: str, root: Path = Path("artifacts")) -> None:
    ds_dir = root / name
    rules_path = ds_dir / "student" / "rules.txt"
    tree_path = ds_dir / "student" / "tree.json"
    if not rules_path.exists():
        print(f"\n{name}: no rules found at {rules_path}")
        return
    print(f"\n=== {name} ===")
    descriptors = _descriptor_map(ds_dir / "descriptors")
    print("\nRules:\n")
    print(rules_path.read_text(encoding="utf-8"))
    if descriptors:
        print("Descriptors:")
        for node_id, desc in sorted(descriptors.items()):
            print(
                f"  node {node_id}: {desc.get('descriptor', '')} "
                f"confidence={desc.get('confidence')} abstained={desc.get('abstained')}"
            )
    else:
        print("Descriptors: none found (descriptor generation may be disabled).")
    if tree_path.exists():
        tree = read_json(tree_path)
        print("Root-to-leaf paths:")
        for path in tree.get("root_to_leaf_paths", []):
            print("  - " + " -> ".join(path))


def main() -> None:
    for name in DATASET_NAMES:
        inspect_dataset(name)


if __name__ == "__main__":
    main()
