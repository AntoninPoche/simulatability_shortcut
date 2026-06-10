from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import torch

from utils.simulatability import AutomatedSimulatability


class RationalePromptSetting(NamedTuple):
    # learning phase configuration
    lp_samples: bool = False
    lp_rationale_justify: bool = False
    lp_rationale_contrastive: bool = False

    # evaluation phase configuration
    include_input_text: bool = True

    # anonymization and masking
    anonymize_classes: bool = False

    def validate(
        self,
        *,
        rationales: list[str] | None,
        contrastives: list[str] | None,
        nb_learning_samples: int,
    ) -> None:
        """
        Validate internal consistency for a prompt setting.

        Arguments:
            rationales: list[str] | None
                Justify rationales for samples. Required when `lp_rationale_justify=True`.
            contrastives: list[str] | None
                Contrastive rationales for samples. Required when `lp_rationale_contrastive=True`.
            nb_learning_samples: int
                Number of learning samples.

        Raises:
            ValueError:
                If the setting is inconsistent or requires missing inputs.
        """
        if self.lp_rationale_justify and self.lp_rationale_contrastive:
            raise ValueError(
                "RationalePromptSetting.lp_rationale_justify and "
                "RationalePromptSetting.lp_rationale_contrastive are mutually exclusive."
            )

        if self.lp_rationale_justify or self.lp_rationale_contrastive:
            if not self.lp_samples:
                raise ValueError(
                    "RationalePromptSetting.lp_rationale_justify or RationalePromptSetting.lp_rationale_contrastive "
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

        if self.lp_rationale_contrastive:
            if contrastives is None:
                raise ValueError(
                    "RationalePromptSetting.lp_rationale_contrastive=True requires `contrastives` "
                    "to be provided to RationalesSimulatability.construct_prompt()."
                )
            if len(contrastives) < nb_learning_samples:
                raise ValueError(
                    f"`contrastives` must have at least `nb_learning_samples` entries. "
                    f"Got {len(contrastives)=} and {nb_learning_samples=}."
                )


class RationalePromptTypes(Enum):
    """
    Named RationalesSimulatability prompt presets.

    Naming convention:
        - `B*`: baselines without rationales.
        - `J*`: justify rationales.
        - `C*`: contrastive rationales.
        - `_anon`: with class anonymization.
        - `_with_lp` / `_without_lp`: whether learning-phase examples are included.
    """

    # Baselines (no rationales)
    L1_baseline_without_lp = RationalePromptSetting()
    L2_baseline_with_lp = RationalePromptSetting(lp_samples=True)

    # Justify rationale settings
    R_justify_with_lp = RationalePromptSetting(
        lp_samples=True,
        lp_rationale_justify=True,
    )

    # Contrastive rationale settings
    RC_contrastive_with_lp = RationalePromptSetting(
        lp_samples=True,
        lp_rationale_contrastive=True,
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
        contrastives: list[str] | None,
    ) -> tuple[str, list[str], list[str]]:
        system_prompt_parts = []

        # ==================================================================================
        # Task description
        task_description = (
            "You are a classifier. For each sample, you have to predict the class. "
        )

        if setting.lp_rationale_justify:
            task_description += (
                "You will have examples of samples, labels, and explanations justifying "
                "the predictions as reference for the task. "
            )
        elif setting.lp_rationale_contrastive:
            task_description += (
                "You will have examples of samples, labels, and contrastive explanations "
                "(why predict this class and not the other) as reference for the task. "
            )
        elif setting.lp_samples:
            task_description += "You will have examples of samples and labels as reference for the task. "

        task_description += (
            "User's prompt will contain an evaluation sample on which you should predict the class. "
            "Only return the class name, no other text."
        )
        system_prompt_parts.append(task_description)

        # ==================================================================================
        # Classes
        display_classes = classes.copy()
        if setting.anonymize_classes:
            display_classes = {i: f"Class_{i}" for i in classes.keys()}

        classes_prompt = (
            f"The classes are: [{', '.join(list(display_classes.values()))}]"
        )
        system_prompt_parts.append(classes_prompt)

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

                # Add contrastive rationale
                if setting.lp_rationale_contrastive and contrastives is not None:
                    rationale = contrastives[i]
                    block.append(f"\tContrastive Explanation: {rationale}")

                learning_phase_blocks.append("\n".join(block))

            system_prompt_parts.append("\n".join(learning_phase_blocks))

        # Concatenate system prompt parts
        system_prompt = "\n\n".join(system_prompt_parts)

        # anonymize classes
        if setting.anonymize_classes:
            for class_id in classes.keys():
                system_prompt = system_prompt.replace(
                    classes[class_id], display_classes[class_id]
                )

        # ==================================================================================
        # Inference (evaluation) phase - user prompts
        user_prompts = []
        for i in range(nb_learning_samples, len(interesting_samples)):
            if setting.include_input_text:
                prompt = "\n".join(
                    [
                        f"Sample_{i}:",
                        f"\tText: {interesting_samples[i]}",
                        "\tLabel: ",
                    ]
                )
            else:
                prompt = "\n".join(
                    [
                        f"Sample_{i}:",
                        "\tLabel: ",
                    ]
                )
            user_prompts.append(prompt)

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
        contrastives: list[str] | None,
        prompt_type: RationalePromptTypes | RationalePromptSetting,
    ) -> None:
        """
        Validate that the selected samples and rationales match the chosen setting.
        """
        setting = RationalesSimulatability._resolve_prompt_setting(prompt_type)
        setting.validate(
            rationales=rationales,
            contrastives=contrastives,
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
        contrastives: list[str] | None = None,
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
            contrastives=contrastives,
            prompt_type=resolved_setting,
        )

        # Extract classes present in predictions
        classes_ids = sorted(corresponding_predictions.unique().tolist())
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
            contrastives=contrastives,
        )
