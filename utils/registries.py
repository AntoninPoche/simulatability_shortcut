"""Lightweight experiment registries.

This module intentionally avoids torch/interpreto/transformers imports so scripts can
validate arguments and compute expected output keys before paying heavy import costs.
"""

from __future__ import annotations

# CLI name -> directory-name prefix mapping for concept artifacts.
CONCEPT_METHOD_NAMES: dict[str, str] = {
    "seminmf": "SemiNMF",
    "ica": "ICA",
    "kmeans": "KMeans",
    "pca": "PCA",
    "svd": "SVD",
    "batchtopk_sae": "BatchTopKSAE",
    "vanilla_sae": "VanillaSAE",
    "neurons": "NeuronsAs",
}

INTERPRETATION_NAMES: dict[str, str] = {
    "topk": "TopKInputs",
    "llm": "LLMLabels",
}

# Valid CLI names for attribution methods. Must match ATTRIBUTION_METHODS in
# utils/attributions.py.
ATTRIBUTION_METHOD_NAMES: tuple[str, ...] = (
    "saliency",
    "integrated_gradients",
    "smooth_grad",
    "square_grad",
    "var_grad",
    "gradient_shap",
    "lime",
    "kernel_shap",
    "occlusion",
    "sobol",
)

# Prompt type abbreviations per explanation family. They match
# enum_member.name.split("_")[0] in the corresponding prompt builder modules.
CONCEPT_PROMPT_ABBREVS: tuple[str, ...] = ("B1", "C1", "B2", "C2", "C3")
RATIONALE_PROMPT_ABBREVS: tuple[str, ...] = ("B1", "B2", "R1")
ATTRIBUTION_PROMPT_ABBREVS: tuple[str, ...] = ("B1", "B2", "A1")
