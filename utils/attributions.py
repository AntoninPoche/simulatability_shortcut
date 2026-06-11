"""Compute and cache token/word-level attributions using interpreto.

This module is analogous to utils/rationales.py but for attribution-based explanations.
It loads a task model, runs an interpreto attribution explainer, and caches the results
so that make_prompts.py can build AttrSim prompts without recomputing.

Cache layout:
    {save_root}/attributions/{method_name}.pt
    Contents: dict[int, dict] mapping sample_id -> {"attributions": Tensor, "elements": list[str], "target": int}
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from tqdm import tqdm

from interpreto import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
    Granularity,
)
from interpreto.attributions.base import AttributionOutput

# ---------------------------------------------------------------------------
# Registry: CLI method name -> interpreto class
# ---------------------------------------------------------------------------
ATTRIBUTION_METHODS = {
    "saliency": Saliency,
    "integrated_gradients": IntegratedGradients,
    "smooth_grad": SmoothGrad,
    "square_grad": SquareGrad,
    "var_grad": VarGrad,
    "gradient_shap": GradientShap,
    "lime": Lime,
    "kernel_shap": KernelShap,
    "occlusion": Occlusion,
    "sobol": Sobol,
}

# Per-method extra kwargs passed to the explainer constructor.
# Empty for now (defaults are fine), but ready for per-method tuning.
ATTRIBUTION_METHOD_KWARGS: dict[str, dict] = {k: {} for k in ATTRIBUTION_METHODS}

# Per-dataset granularity override.  Default is WORD for text classification.
# Use SENTENCE for datasets where individual tokens are not meaningful (e.g. IMDB reviews).
DATASET_GRANULARITY: dict[str, Granularity] = {
    # All current datasets use word-level:
    # "google-research-datasets/go_emotions": Granularity.WORD,
    # "dair-ai/emotion": Granularity.WORD,
    # "Hate-speech-CNERG/hatexplain": Granularity.WORD,
    # "LabHC/bias_in_bios": Granularity.WORD,
    # Add future datasets that need sentence-level here:
    "stanfordnlp/imdb": Granularity.SENTENCE,
}
DEFAULT_GRANULARITY = Granularity.WORD


def _get_granularity(dataset_name: str | None) -> Granularity:
    """Resolve the granularity for a given dataset (falls back to WORD)."""
    if dataset_name is None:
        return DEFAULT_GRANULARITY
    return DATASET_GRANULARITY.get(dataset_name, DEFAULT_GRANULARITY)


def _serialize_attribution(attr: AttributionOutput, sample_id: int) -> dict:
    """Extract the minimal cacheable fields from an AttributionOutput."""
    return {
        "sample_id": sample_id,
        "attributions": attr.attributions.cpu(),
        "elements": list(attr.elements)
        if not isinstance(attr.elements, list)
        else attr.elements,
        "target": int(attr.targets.item())
        if attr.targets.numel() == 1
        else attr.targets.cpu().tolist(),
    }


def _reconstruct_attribution(record: dict) -> AttributionOutput:
    """Reconstruct an AttributionOutput from cached minimal fields."""
    target = record["target"]
    if isinstance(target, int):
        targets = torch.tensor([target])
    else:
        targets = torch.tensor(target)

    return AttributionOutput(
        attributions=record["attributions"],
        elements=record["elements"],
        model_inputs_to_explain={},  # not needed for prompt construction
        targets=targets,
        model_task="classification",  # ModelTask enum, but string works for our use
    )


def load_or_compute_attributions(
    *,
    model_name: str,
    inputs: list[str],
    predictions: torch.Tensor,
    sample_ids: list[int],
    method: str,
    save_root: Path,
    device: str,
    batch_size: int = 4,
    dataset_name: str | None = None,
) -> dict[int, AttributionOutput]:
    """
    Compute or load cached attributions for the given samples.

    Arguments:
        model_name: HuggingFace model name or local path (classification model).
        inputs: Text inputs aligned with sample_ids.
        predictions: Model predictions aligned with sample_ids.
        sample_ids: Global test-set indices for each input.
        method: Key in ATTRIBUTION_METHODS (e.g. "saliency").
        save_root: Root directory for caching artifacts.
        device: Torch device string.
        batch_size: Batch size for the attribution explainer.
        dataset_name: Optional dataset name (used for granularity lookup).

    Returns:
        dict mapping sample_id -> reconstructed AttributionOutput.
    """
    if len(sample_ids) != len(inputs):
        raise ValueError(
            f"`sample_ids` must match `inputs` length. Got {len(sample_ids)=} and {len(inputs)=}."
        )
    if method not in ATTRIBUTION_METHODS:
        raise ValueError(
            f"Unknown attribution method '{method}'. Choose from: {list(ATTRIBUTION_METHODS.keys())}"
        )

    # Cache path
    cache_dir = save_root / "attributions"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = cache_dir / f"{method}.pt"

    # Load existing cache
    cached: dict[int, dict] = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")

    # Determine which samples still need computation
    missing_ids = [sid for sid in sample_ids if sid not in cached]

    if missing_ids:
        # Build offset map: sample_id -> position in the input lists
        sid_to_offset = {sid: offset for offset, sid in enumerate(sample_ids)}

        # Load task model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_device = torch.device(device)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(model_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Instantiate attribution explainer
        granularity = _get_granularity(dataset_name)
        method_cls = ATTRIBUTION_METHODS[method]
        method_kwargs = ATTRIBUTION_METHOD_KWARGS.get(method, {})
        explainer = method_cls(
            model,
            tokenizer,
            batch_size=batch_size,
            granularity=granularity,
            device=model_device,
            **method_kwargs,
        )

        # Compute attributions for missing samples in batches
        for batch_start in tqdm(
            range(0, len(missing_ids), batch_size),
            desc=f"Attributions ({method})",
        ):
            batch_sids = missing_ids[batch_start : batch_start + batch_size]
            batch_inputs = [inputs[sid_to_offset[sid]] for sid in batch_sids]
            batch_preds = torch.tensor(
                [int(predictions[sid_to_offset[sid]]) for sid in batch_sids]
            )

            # Explain the predicted class for each sample.
            # targets shape: (batch, 1) — one target class per sample.
            targets = batch_preds.unsqueeze(1)
            attr_outputs: list[AttributionOutput] = explainer.explain(
                batch_inputs, targets=targets
            )

            for sid, attr_out in zip(batch_sids, attr_outputs, strict=True):
                cached[sid] = _serialize_attribution(attr_out, sid)

        # Persist updated cache
        torch.save(cached, cache_path)

        # Release GPU memory
        del explainer, model, tokenizer
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

    # Return only the requested samples as reconstructed AttributionOutput objects
    return {sid: _reconstruct_attribution(cached[sid]) for sid in sample_ids}


def group_attributions_by_seed(
    attributions: dict[int, AttributionOutput],
    seed_indices: dict[int, list[int]],
) -> dict[int, list[AttributionOutput]]:
    """
    Group attribution artifacts by seed, preserving sample order.

    Arguments:
        attributions: Mapping from sample_id to AttributionOutput.
        seed_indices: Mapping from seed to ordered list of sample_ids.

    Returns:
        dict mapping seed -> list[AttributionOutput] aligned with the seed's sample order.
    """
    result: dict[int, list[AttributionOutput]] = {}
    for seed, indices in seed_indices.items():
        result[seed] = [attributions[idx] for idx in indices]
    return result
