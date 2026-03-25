import base64
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

from config import (
    IMAGENETTE_CLASSES,
    VLM_COMPONENTS_TO_DESCRIBE,
    VLM_DESCRIPTOR_ANALYSIS_PATH,
    VLM_DESCRIPTOR_CONFIDENCE_THRESHOLD,
    VLM_DESCRIPTOR_JSON_PATH,
    VLM_DESCRIPTOR_DIR,
    VLM_MODEL,
    VLM_PROVIDER,
    VLM_REQUEST_TIMEOUT_SEC,
    VLM_TOP_SAMPLES_PER_COMPONENT,
)


def _tensor_to_base64_png(image_tensor):
    img = image_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]

    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    arr = img.numpy().astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)

    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _denormalize_tensor(image_tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = image_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]

    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    arr = img.numpy().astype(np.float32)
    arr = (arr * std) + mean
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _call_openai_vlm(prompt_text, base64_images, model=VLM_MODEL):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not set; skipping VLM call."

    input_content = [{"type": "input_text", "text": prompt_text}]
    for b64 in base64_images:
        input_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": input_content}],
        "text": {"format": {"type": "json_object"}},
    }

    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=VLM_REQUEST_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="ignore")
        return None, f"OpenAI API error ({exc.code}): {err_text[:400]}"
    except URLError as exc:
        return None, f"OpenAI API network error: {exc.reason}"
    output_text = body.get("output_text", "")
    if output_text:
        return output_text, None

    output = body.get("output", [])
    for item in output:
        for part in item.get("content", []):
            txt = part.get("text")
            if txt:
                return txt, None

    return None, "No textual output in VLM response."


def _build_prompt(component_index, explained_var, sample_labels):
    label_counts = {}
    for label in sample_labels:
        label_counts[int(label)] = label_counts.get(int(label), 0) + 1

    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    label_summary = ", ".join(
        [f"{IMAGENETTE_CLASSES[label]}:{count}" for label, count in sorted_counts]
    )

    return (
        "You are helping with interpretable image classification. "
        "I am giving you representative images for one PCA component from a ResNet feature space. "
        "Return strict JSON with keys: selected_descriptor (string), selected_confidence (0-1 float), "
        "candidates (array of objects with keys descriptor, confidence, relevance_score), and rationale (string). "
        "Descriptors must be concise noun phrases that reflect recurring visual patterns (e.g., texture, shape, object part). "
        f"Component index: {component_index}. Explained variance ratio: {explained_var:.6f}. "
        f"Class mix among representative samples: {label_summary}. "
        "Prefer descriptors that could be used as feature names in a decision tree."
    )


def _safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _select_candidate(parsed, threshold):
    if not parsed:
        return None

    candidates = parsed.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    filtered = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        conf = float(c.get("confidence", 0.0) or 0.0)
        relevance = float(c.get("relevance_score", 0.0) or 0.0)
        if conf >= threshold:
            filtered.append((conf * 0.7 + relevance * 0.3, c))

    if filtered:
        filtered.sort(key=lambda x: x[0], reverse=True)
        return filtered[0][1]

    selected_descriptor = parsed.get("selected_descriptor")
    selected_conf = float(parsed.get("selected_confidence", 0.0) or 0.0)
    if isinstance(selected_descriptor, str) and selected_conf >= threshold:
        return {"descriptor": selected_descriptor, "confidence": selected_conf, "relevance_score": selected_conf}

    return None


def generate_vlm_component_descriptors(
    train_pca_features,
    pca_model,
    train_dataset,
    used_feature_indices,
    output_path=VLM_DESCRIPTOR_JSON_PATH,
    max_components=VLM_COMPONENTS_TO_DESCRIBE,
    top_samples=VLM_TOP_SAMPLES_PER_COMPONENT,
    confidence_threshold=VLM_DESCRIPTOR_CONFIDENCE_THRESHOLD,
):
    Path(VLM_DESCRIPTOR_DIR).mkdir(parents=True, exist_ok=True)

    used_indices = list(used_feature_indices)[:max_components]
    descriptor_records = []

    for pc_idx in used_indices:
        component_scores = np.abs(train_pca_features[:, pc_idx])
        top_idx = np.argsort(component_scores)[-top_samples:][::-1]

        sample_images = []
        sample_labels = []
        sample_ids = []

        for idx in top_idx:
            image_tensor, label = train_dataset[idx]
            denorm = _denormalize_tensor(image_tensor)
            sample_images.append(_tensor_to_base64_png(denorm))
            sample_labels.append(label)
            sample_ids.append(int(idx))

        prompt_text = _build_prompt(
            component_index=int(pc_idx),
            explained_var=float(pca_model.explained_variance_ratio_[pc_idx]),
            sample_labels=sample_labels,
        )

        raw_response = None
        error = None

        if VLM_PROVIDER == "openai":
            raw_response, error = _call_openai_vlm(prompt_text, sample_images, model=VLM_MODEL)
        else:
            error = f"Unsupported VLM provider: {VLM_PROVIDER}"

        parsed = _safe_json_loads(raw_response) if raw_response else None
        selected = _select_candidate(parsed, confidence_threshold)

        descriptor_records.append(
            {
                "component_index": int(pc_idx),
                "explained_variance_ratio": float(pca_model.explained_variance_ratio_[pc_idx]),
                "sample_indices": sample_ids,
                "sample_labels": [int(x) for x in sample_labels],
                "prompt": prompt_text,
                "raw_response": raw_response,
                "error": error,
                "parsed": parsed,
                "selected_descriptor": selected["descriptor"] if selected else None,
                "selected_confidence": float(selected["confidence"]) if selected else None,
                "selected_relevance": float(selected.get("relevance_score", 0.0)) if selected else None,
            }
        )

        print(
            f"[VLM] PC {pc_idx}: "
            f"descriptor={descriptor_records[-1]['selected_descriptor']} "
            f"conf={descriptor_records[-1]['selected_confidence']} "
            f"error={error}"
        )

    with open(output_path, "w") as f:
        json.dump(descriptor_records, f, indent=2)

    print(f"[VLM] Saved descriptor summary to {output_path}")
    return descriptor_records


def build_descriptor_feature_names(n_components, descriptor_records):
    feature_names = [f"PC {i}" for i in range(n_components)]
    for rec in descriptor_records:
        idx = rec["component_index"]
        desc = rec.get("selected_descriptor")
        conf = rec.get("selected_confidence")
        if isinstance(desc, str) and desc.strip():
            if conf is not None:
                feature_names[idx] = f"{desc.strip()} ({conf:.2f})"
            else:
                feature_names[idx] = desc.strip()
    return feature_names


def analyze_descriptor_quality(descriptor_records, output_path=VLM_DESCRIPTOR_ANALYSIS_PATH):
    total = len(descriptor_records)
    with_descriptor = [r for r in descriptor_records if r.get("selected_descriptor")]
    with_response = [r for r in descriptor_records if r.get("raw_response")]
    errored = [r for r in descriptor_records if r.get("error")]

    confidences = [
        float(r["selected_confidence"])
        for r in with_descriptor
        if r.get("selected_confidence") is not None
    ]

    analysis = {
        "total_components_attempted": total,
        "components_with_vlm_response": len(with_response),
        "components_with_selected_descriptor": len(with_descriptor),
        "components_with_errors": len(errored),
        "descriptor_coverage": (len(with_descriptor) / total) if total else 0.0,
        "mean_selected_confidence": float(np.mean(confidences)) if confidences else None,
        "min_selected_confidence": float(np.min(confidences)) if confidences else None,
        "max_selected_confidence": float(np.max(confidences)) if confidences else None,
    }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"[VLM] Saved descriptor analysis to {output_path}")
    return analysis
