from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import torch

from interpreto.attributions.base import AttributionOutput
from utils.simulatability import AutomatedSimulatability


class PromptSetting(NamedTuple):
    """
    Low-level configuration of a AttrSim prompt.

    Each flag enables one prompt block. `PromptTypes` exposes the common presets used in papers and
    tests, while direct `PromptSetting(...)` instances let advanced users define custom ablations.

    Attributes:
        lp_samples: bool
            Include learning-phase examples in the shared system prompt.
        lp_attributions: bool
            Add attribution explanations for each learning-phase example.
        anonymize_classes: bool
            Replace user-facing class names with `Class_i`.
            Preventing the LLM from using knowledge on classes names.
    """

    lp_samples: bool = False
    lp_attributions: bool = False
    anonymize_classes: bool = False
    attribution_top_k: int = 6
    attribution_onlypositivevalues: bool = True

    def validate(
        self,
        *,
        labels: torch.Tensor | list[int] | None,
    ) -> None:
        """
        Validate internal consistency for a prompt setting.

        Arguments:
            labels: torch.Tensor | list[int] | None
                Gold labels aligned with the selected samples. Kept for API symmetry.

        Raises:
            ValueError:
                If the setting is inconsistent or requires missing inputs.
        """
        if not (self.lp_samples) and self.lp_attributions:
            raise ValueError("PromptSetting.lp_attributions requires `lp_samples=True`.")

        if self.attribution_top_k <= 0:
            raise ValueError("PromptSetting.attribution_top_k must be strictly positive.")


class PromptTypes(Enum):
    """
    Named AttrSim prompt presets.

    Naming convention:
        - `B*`: baselines without attribution explanations.
        - `A*`: standard attribution explanations during learning phase.
        - `with_lp` / `without_lp`: whether learning-phase examples are included.

    Each enum value is a `PromptSetting`. Use the enum for standard experiments and direct
    `PromptSetting(...)` values for custom studies.
    """

    B1_baseline_without_lp = PromptSetting()
    B2_baseline_with_lp = PromptSetting(lp_samples=True)

    A1_attribution_with_lp = PromptSetting(lp_samples=True, lp_attributions=True)


class AttrSim(AutomatedSimulatability):
    """
    AttrSim prompt builder for attribution-based automated simulatability.

    AttrSim measures whether attribution explanations help a meta-predictor reproduce a classifier's
    outputs. In this module, `AttrSim` is responsible only for AttrSim-specific prompt
    design and validation. It does not compute model predictions, token/word/sentence-level attributions, or call the
    LLM on its own.

    Therefore, users need to compute model predictions and attribution explanations beforehand.

    Typical workflow:
        1. Instantiate `AttrSim(classes=...)`.
        2. Call `select_examples(...)` on precomputed inputs, labels, and model predictions.
        3. Use a fitted attribution explainer upstream to build the explanation artifacts required by
           the chosen setting.
        4. Call `construct_prompt(...)`.
        5. Run the prompts through your LLM interface outside this class.
        6. Compute responses with `llm_interface.batch_generate(...)`.
        7. Score the returned responses with `score_from_responses(...)`.

    Arguments:
        classes: list[str]
            Display names for class ids. Inherited from `AutomatedSimulatability`; `classes[i]`
            must match class id `i`.

    Attributes:
        classes: list[str]
            Display names for class ids.
        prompt_types: type[PromptTypes]
            Preset prompt configurations shipped with AttrSim.
            These are prompt settings that can be passed to `AttrSim.construct_prompt()`.
    """

    prompt_types: type[PromptTypes] = PromptTypes

    @staticmethod
    def _resolve_prompt_setting(setting: PromptTypes | PromptSetting) -> PromptSetting:
        """
        Normalize a prompt setting input to a concrete `PromptSetting`.

        Args:
            setting: Either a `PromptTypes` enum preset or a direct `PromptSetting`.

        Returns:
            PromptSetting: Resolved prompt setting.
        """
        return setting.value if isinstance(setting, PromptTypes) else setting

    @staticmethod
    def _get_attr_vector(
        attribution_output: AttributionOutput,
        class_index: int,
    ) -> torch.Tensor:
        """
        Extract the attribution vector associated with one class.

        Handles three accepted attribution layouts:
            - `(l,)`: single class vector.
            - `(1, l)`: singleton class axis.
            - `(c, l)`: class-wise vectors, indexed by `class_index`.

        Args:
            attribution_output: Attribution container for one sample.
            class_index: Class index whose vector should be extracted.

        Returns:
            torch.Tensor: Attribution vector of shape `(l,)`.

        Raises:
            ValueError: If `class_index` is out of bounds for class-wise attributions.
        """
        attributions = attribution_output.attributions
        if attributions.ndim == 1:
            return attributions
        elif attributions.ndim == 2 and attributions.shape[0] == 1:
            return attributions[0]

        if class_index >= attributions.shape[0]:
            raise ValueError(
                "Attribution tensor does not contain enough class-wise rows to format this sample. "
                f"Requested class index {class_index}, but attributions has shape {tuple(attributions.shape)}."
            )
        return attributions[class_index]

    @staticmethod
    def _format_attr_vector(
        elements: list[str] | torch.Tensor,
        attr_vector: torch.Tensor,
        top_k: int = 6,
        *,
        select_by_abs: bool = True,
    ) -> str:
        """
        Format one attribution vector as a `{token: score}` string.

        Scores are first normalized over the full sentence using L1 normalization
        (`attr / sum(abs(attr))`). Then top-k elements are selected according to
        `only_positive_values`.

        Args:
            elements: Tokens/elements aligned with attribution positions.
            attr_vector: Attribution scores `(l,)`.
            top_k: Number of elements to include.
            only_positive_values:
                - True: select top positive normalized scores.
                - False: select by absolute normalized score.

        Returns:
            str: Rendered attribution dictionary string.
        """
        if isinstance(elements, torch.Tensor):
            elements = [str(e.item()) for e in elements]
        else:
            elements = [str(e) for e in elements]

        normalized_attr = AttrSim._normalize_attr_vector(attr_vector)
        top_k = min(top_k, normalized_attr.shape[-1])
        ranking_attr = normalized_attr.abs() if select_by_abs else normalized_attr
        top_indices = torch.topk(ranking_attr, k=top_k).indices.tolist()

        pieces = []
        for idx in top_indices:
            token = elements[idx] if idx < len(elements) else f"tok_{idx}"
            pieces.append(f"{token}: {normalized_attr[idx].item():+.3f}")
        return "{" + ", ".join(pieces) + "}"

    @staticmethod
    def _normalize_attr_vector(attr_vector: torch.Tensor) -> torch.Tensor:
        """
        Normalize an attribution vector with sentence-level L1 normalization.

        Args:
            attr_vector: Raw attribution vector `(l,)`.

        Returns:
            torch.Tensor: Normalized vector. Returns all-zeros if denominator is zero.
        """
        denom = attr_vector.abs().sum()
        if torch.isclose(denom, torch.tensor(0.0, device=attr_vector.device, dtype=attr_vector.dtype)):
            return torch.zeros_like(attr_vector)
        return attr_vector / denom

    @staticmethod
    def _format_attribution_for_pred(
        attribution_output: AttributionOutput,
        pred_index: int,
        top_k: int = 6,
        *,
        select_by_abs: bool = True,
    ) -> str:
        """
        Format attributions for one predicted class.

        Args:
            attribution_output: Attribution container for one sample.
            pred_index: Predicted class index.
            top_k: Number of elements to include.
            only_positive_values: Top-k selection mode (positive-only vs absolute).

        Returns:
            str: Rendered attribution dictionary string.
        """
        pred_attr = AttrSim._get_attr_vector(attribution_output, pred_index)
        return AttrSim._format_attr_vector(
            attribution_output.elements,
            pred_attr,
            top_k=top_k,
            select_by_abs=select_by_abs,
        )

    def construct_prompt(  # type: ignore
        self,
        setting: PromptTypes | PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        *,
        corresponding_attribution: list[AttributionOutput],
        class_ids: list[int] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Build AttrSim system and user prompts from selected examples.

        Args:
            setting: Prompt preset (`PromptTypes`) or explicit `PromptSetting`.
            interesting_samples: Selected texts used for LP + evaluation phases.
            corresponding_predictions: Model predictions aligned with samples.
            corresponding_labels: Ground-truth labels aligned with samples.
            nb_learning_samples: Number of first samples used in LP context.
            corresponding_attribution: Attribution outputs aligned with samples.

        Returns:
            tuple[str, list[str], list[str]]:
                - system prompt containing LP examples and explanations,
                - user prompts for evaluation samples,
                - expected model prediction labels for evaluation samples.
        """
        setting = self._resolve_prompt_setting(setting)
        self._check_input_settings_correspondence(
            setting=setting,
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            corresponding_attribution=corresponding_attribution,
        )

        classes_ids = (
            sorted(int(class_id) for class_id in class_ids)
            if class_ids is not None
            else sorted(corresponding_predictions.unique().tolist())
        )
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}

        if setting.anonymize_classes:
            classes = {i: f"Class_{i}" for i in classes.keys()}

        # Task description harmonized with ConSim wording so that B1/B2 produce
        # byte-identical system prompts across concepts/rationales/attributions.
        task_description_prompt = "You are a classifier. Your task is to assign a label to the evaluation sample. "
        if setting.lp_samples:
            if setting.lp_attributions:
                task_description_prompt += (
                    "You will have examples of samples, labels, and attribution explanations as reference to learn the task. "
                )
            else:
                task_description_prompt += (
                    "You will have examples of samples and labels as reference to learn the task. "
                )
        if setting.lp_attributions:
            task_description_prompt += (
                "For each sample, the attributions are tokens with their importance scores, where positive scores support the prediction and negative scores oppose it. "
            )
        task_description_prompt += "User's prompt will contain an evaluation sample on which you should predict the class. Only return the class name, no other text."

        system_prompt_parts = [
            task_description_prompt,
            f"The classes are: [{', '.join(list(classes.values()))}]",
        ]

        if setting.lp_samples:
            lp_blocks = []
            for i in range(nb_learning_samples):
                pred_index = int(corresponding_predictions[i])
                lp_block = [
                    f"Sample_{i}:",
                    f"\tText: {interesting_samples[i]}",
                    f"\tLabel: {classes[pred_index]}",
                ]
                if setting.lp_attributions:
                    formatted_attr = self._format_attribution_for_pred(
                        corresponding_attribution[i],
                        pred_index,
                        top_k=setting.attribution_top_k,
                        select_by_abs=setting.attribution_onlypositivevalues,
                    )

                    lp_block.append(f"\tAttributions: {formatted_attr}")

                lp_blocks.append("\n".join(lp_block))

            system_prompt_parts.append("\n".join(lp_blocks))

        system_prompt = "\n\n".join(system_prompt_parts)

        user_prompts: list[str] = []
        model_predictions: list[str] = []
        for i in range(nb_learning_samples, len(interesting_samples)):
            user_prompts.append(f"Evaluation sample:\n\tText: {interesting_samples[i]}\n\tLabel: ")
            model_predictions.append(classes[int(corresponding_predictions[i])])

        return system_prompt, user_prompts, model_predictions

    def _check_input_settings_correspondence(
        self,
        setting: PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        corresponding_attribution: list[AttributionOutput] | None,
    ) -> None:
        """
        Validate consistency between inputs and selected prompt setting.

        This validates:
            - lengths alignment across samples/predictions/labels/attributions,
            - LP size constraints,
            - required attribution presence for attribution-based settings.

        Args:
            setting: Resolved prompt setting.
            interesting_samples: Selected texts.
            corresponding_predictions: Predictions aligned with texts.
            corresponding_labels: Labels aligned with texts.
            nb_learning_samples: Number of LP samples.
            corresponding_attribution: Optional attributions aligned with texts.

        Raises:
            ValueError: If an inconsistency is detected.
        """
        setting.validate(labels=corresponding_labels)

        if len(corresponding_predictions) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_predictions` must have the same length.")

        if len(corresponding_labels) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_labels` must have the same length.")

        if nb_learning_samples >= len(interesting_samples):
            raise ValueError("`nb_learning_samples` must be smaller than number of provided samples.")

        if corresponding_attribution is None:
            if setting.lp_attributions:
                raise ValueError(
                    "`corresponding_attribution` is required when using attribution-based learning prompts."
                )
            return

        if len(corresponding_attribution) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_attribution` must have the same length.")
