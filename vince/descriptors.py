from __future__ import annotations

import base64
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from PIL import Image
from sklearn.tree import _tree

from .data import sample_path, sample_pil_image
from .utils import ensure_dir, read_openai_api_key, write_json

# Descriptor labels are post-hoc semantic hypotheses. They are not proven causal
# explanations and are not guaranteed to be faithful to the teacher model.

SCHEMA_HINT = {
    "descriptor": "short human-readable visual concept",
    "left_description": "what images on the left side tend to show",
    "right_description": "what images on the right side tend to show",
    "confidence": 0.0,
    "failure_mode": "empty string if none, otherwise why the descriptor may be unreliable",
}


ImageSource = Union[Path, Image.Image]


def _image_to_data_url(image: ImageSource) -> str:
    if isinstance(image, Path):
        with image.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
    else:
        import io

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _save_grid(images: list[ImageSource], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    if not images:
        write_json(out_path.with_suffix(".json"), {"warning": "no representative images available"})
        return
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes = [axes]
    for i, (ax, image) in enumerate(zip(axes, images)):
        if isinstance(image, Path):
            ax.imshow(Image.open(image).convert("RGB"))
            ax.set_title(image.name[:20])
        else:
            ax.imshow(image.convert("RGB"))
            ax.set_title(f"sample {i}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _node_sides(tree_model, X: np.ndarray, node_id: int) -> tuple[np.ndarray, np.ndarray]:
    tree = tree_model.tree_
    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    if feature == _tree.TREE_UNDEFINED:
        return np.array([], dtype=int), np.array([], dtype=int)
    reaches = tree_model.decision_path(X)[:, node_id].toarray().ravel().astype(bool)
    left = np.where(reaches & (X[:, feature] <= threshold))[0]
    right = np.where(reaches & (X[:, feature] > threshold))[0]
    return left, right


def _representative_images(dataset, indices: np.ndarray, max_images: int) -> list[ImageSource]:
    images: list[ImageSource] = []
    for i in indices[:max_images]:
        path = sample_path(dataset, int(i))
        if path and path.exists():
            images.append(path)
            continue
        image = sample_pil_image(dataset, int(i))
        if image is not None:
            images.append(image)
    return images


def _call_openai(client: OpenAI, model: str, left: list[ImageSource], right: list[ImageSource], variant: int) -> dict:
    prompt = (
        "Return strict JSON only, following this schema: " + json.dumps(SCHEMA_HINT) + "\n"
        "Compare the left-branch images with the right-branch images from a decision-tree split. "
        "Give a short visual descriptor hypothesis. Do not claim causality or faithfulness."
    )
    if variant == 1:
        prompt += " Focus on simple visible attributes such as object, color, texture, pose, or background."
    content = [{"type": "input_text", "text": prompt}]
    for label, paths in [("LEFT", left), ("RIGHT", right)]:
        content.append({"type": "input_text", "text": label})
        for image in paths:
            content.append({"type": "input_image", "image_url": _image_to_data_url(image)})
    response = client.responses.create(model=model, input=[{"role": "user", "content": content}], temperature=0)
    text = response.output_text
    return {"raw_text": text, "parsed": json.loads(text)}


def generate_descriptors(dataset_name: str, dataset, X_train: np.ndarray, tree_model, out_dir: Path, config) -> None:
    ensure_dir(out_dir)
    client = OpenAI(api_key=read_openai_api_key())
    tree = tree_model.tree_
    described = 0
    stability = []
    for node_id in range(tree.node_count):
        if described >= config.max_nodes_to_describe:
            break
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            continue
        node_dir = ensure_dir(out_dir / f"node_{node_id}")
        left_idx, right_idx = _node_sides(tree_model, X_train, node_id)
        left_paths = _representative_images(dataset, left_idx, config.max_images_per_side)
        right_paths = _representative_images(dataset, right_idx, config.max_images_per_side)
        _save_grid(left_paths, node_dir / "left_grid.png", f"{dataset_name} node {node_id} left")
        _save_grid(right_paths, node_dir / "right_grid.png", f"{dataset_name} node {node_id} right")
        try:
            first = _call_openai(client, config.openai_model, left_paths, right_paths, variant=0)
            second = _call_openai(client, config.openai_model, left_paths, right_paths, variant=1)
            write_json(node_dir / "raw_response_1.json", first)
            write_json(node_dir / "raw_response_2.json", second)
            parsed = first["parsed"]
            confidence = float(parsed.get("confidence", 0.0))
            parsed["node_id"] = node_id
            parsed["abstained"] = confidence < config.descriptor_confidence_threshold
            write_json(node_dir / "node_descriptor.json", parsed)
            ratio = SequenceMatcher(None, str(first["parsed"].get("descriptor", "")), str(second["parsed"].get("descriptor", ""))).ratio()
            stability.append({"node_id": node_id, "descriptor_similarity": ratio})
        except Exception as exc:
            write_json(node_dir / "openai_error.json", {"node_id": node_id, "error_type": type(exc).__name__, "error": str(exc)})
        described += 1
    write_json(out_dir / "descriptor_stability.json", {"items": stability})
