from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple

import torch

from utils.simulatability import AutomatedSimulatability


class RationalePromptSetting(NamedTuple):
    # learning phase configuration
    lp_samples: bool = False
    lp_rationale_justify: bool = False

    # anonymization and masking
    anonymize_classes: bool = False

    def validate(
        self,
        *,
        rationales: list[str] | None,
        nb_learning_samples: int,
    ) -> None:
        """
        Validate internal consistency for a prompt setting.

        Arguments:
            rationales: list[str] | None
                Justify rationales for samples. Required when `lp_rationale_justify=True`.
            nb_learning_samples: int
                Number of learning samples.

        Raises:
            ValueError:
                If the setting is inconsistent or requires missing inputs.
        """
        if self.lp_rationale_justify:
            if not self.lp_samples:
                raise ValueError(
                    "RationalePromptSetting.lp_rationale_justify "
                    "requires `lp_samples=True`."
                )

        if self.lp_rationale_justify:
            if rationales is None:
                raise ValueError(
                    "RationalePromptSetting.lp_rationale_justify=True requires `rationales` "
                    "to be provided to RationalesSimulatability.construct_prompt()."
                )
            if len(rationales) < nb_learning_samples:
                raise ValueError(
                    f"`rationales` must have at least `nb_learning_samples` entries. "
                    f"Got {len(rationales)=} and {nb_learning_samples=}."
                )


class RationalePromptTypes(Enum):
    """
    Named RationalesSimulatability prompt presets.

    Naming convention:
        - `B*`: baselines without rationales.
        - `R*`: justify rationales.
        - `_anon`: with class anonymization.
        - `_with_lp` / `_without_lp`: whether learning-phase examples are included.
    """

    # Baselines (no rationales)
    B1_baseline_without_lp = RationalePromptSetting()
    B2_baseline_with_lp = RationalePromptSetting(lp_samples=True)

    # Justify rationale settings
    R1_justify_with_lp = RationalePromptSetting(
        lp_samples=True,
        lp_rationale_justify=True,
    )


class RationalesSimulatability(AutomatedSimulatability):
    prompt_types: type[RationalePromptTypes] = RationalePromptTypes

    @staticmethod
    def _resolve_prompt_setting(
        prompt_type: RationalePromptTypes | RationalePromptSetting,
    ) -> RationalePromptSetting:
        if isinstance(prompt_type, RationalePromptTypes):
            return prompt_type.value
        return prompt_type

    @staticmethod
    def _setting_to_prompt(
        setting: RationalePromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        classes: dict[int, str],
        rationales: list[str] | None,
    ) -> tuple[str, list[str], list[str]]:
        system_prompt_parts = []

        # ==================================================================================
        # Task description (harmonized with ConSim wording: B1/B2 produce byte-identical
        # text across concepts/rationales/attributions families).
        task_description_prompt = "You are a classifier. Your task is to assign a label to the evaluation sample. "
        if setting.lp_samples:
            if setting.lp_rationale_justify:
                task_description_prompt += (
                    "You will have examples of samples, labels, and explanations justifying "
                    "the predictions as reference to learn the task. "
                )
            else:
                task_description_prompt += (
                    "You will have examples of samples and labels as reference to learn the task. "
                )
        if setting.lp_rationale_justify:
            task_description_prompt += (
                "For each sample, the explanation justifies why the predicted label is correct. "
            )
        task_description_prompt += "User's prompt will contain an evaluation sample on which you should predict the class. Only return the class name, no other text."
        system_prompt_parts.append(task_description_prompt)

        # ==================================================================================
        # Classes
        display_classes = classes.copy()
        if setting.anonymize_classes:
            display_classes = {i: f"Class_{i}" for i in classes.keys()}

        system_prompt_parts.append(
            f"The classes are: [{', '.join(list(display_classes.values()))}]"
        )

        # Learning phase
        if setting.lp_samples:
            learning_phase_blocks = []

            for i in range(nb_learning_samples):
                pred_id = int(corresponding_predictions[i])
                label_name = display_classes[pred_id]

                block = [
                    f"Sample_{i}:",
                    f"\tText: {interesting_samples[i]}",
                    f"\tLabel: {label_name}",
                ]

                # Add justify rationale
                if setting.lp_rationale_justify and rationales is not None:
                    rationale = rationales[i]
                    block.append(f"\tExplanation: {rationale}")

                learning_phase_blocks.append("\n".join(block))

            system_prompt_parts.append("\n".join(learning_phase_blocks))

        # Concatenate system prompt parts
        system_prompt = "\n\n".join(system_prompt_parts)

        # Anonymize lingering class-name occurrences inside LP example texts and rationale
        # text. Word-boundary regex (case-insensitive) so that substrings inside other words
        # are NOT touched (e.g. "position" must not become "Class_1ition" when class="pos").
        if setting.anonymize_classes:
            for class_id, class_name in classes.items():
                pattern = re.compile(
                    rf"(?<!\w){re.escape(class_name)}(?!\w)", re.IGNORECASE
                )
                system_prompt = pattern.sub(display_classes[class_id], system_prompt)

        # ==================================================================================
        # Inference (evaluation) phase - user prompts
        # Harmonized with ConSim/AttrSim format: one "Evaluation sample:" block per sample.
        user_prompts = [
            "\n".join(
                [
                    "Evaluation sample:",
                    f"\tText: {interesting_samples[i]}",
                    "\tLabel: ",
                ]
            )
            for i in range(nb_learning_samples, len(interesting_samples))
        ]

        # Model predictions (expected answers)
        literal_model_predictions = [
            display_classes[int(corresponding_predictions[i])]
            for i in range(nb_learning_samples, len(interesting_samples))
        ]

        return system_prompt, user_prompts, literal_model_predictions

    def _check_input_settings_correspondence(
        self,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        rationales: list[str] | None,
        prompt_type: RationalePromptTypes | RationalePromptSetting,
    ) -> None:
        """
        Validate that the selected samples and rationales match the chosen setting.
        """
        setting = RationalesSimulatability._resolve_prompt_setting(prompt_type)
        setting.validate(
            rationales=rationales,
            nb_learning_samples=nb_learning_samples,
        )

        # Length checks
        if nb_learning_samples >= len(interesting_samples):
            raise ValueError(
                f"`nb_learning_samples` must be smaller than the number of samples. "
                f"Got {nb_learning_samples=} and {len(interesting_samples)=}."
            )

        if len(corresponding_predictions) != len(interesting_samples):
            raise ValueError(
                f"`corresponding_predictions` and `interesting_samples` must have the same length. "
                f"Got {len(corresponding_predictions)=} and {len(interesting_samples)=}."
            )

        if len(corresponding_labels) != len(interesting_samples):
            raise ValueError(
                f"`corresponding_labels` and `interesting_samples` must have the same length. "
                f"Got {len(corresponding_labels)=} and {len(interesting_samples)=}."
            )

    def construct_prompt(
        self,
        setting: RationalePromptTypes | RationalePromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        *,
        rationales: list[str] | None = None,
        class_ids: list[int] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Build the prompts needed to run a RationalesSimulatability evaluation.
        """
        resolved_setting = RationalesSimulatability._resolve_prompt_setting(setting)

        # Validate inputs
        self._check_input_settings_correspondence(
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            rationales=rationales,
            prompt_type=resolved_setting,
        )

        # Render the active class subset, not only classes present in predictions.
        classes_ids = (
            sorted(int(class_id) for class_id in class_ids)
            if class_ids is not None
            else sorted(corresponding_predictions.unique().tolist())
        )
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}

        # Render prompts
        return RationalesSimulatability._setting_to_prompt(
            setting=resolved_setting,
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            classes=classes,
            rationales=rationales,
        )
