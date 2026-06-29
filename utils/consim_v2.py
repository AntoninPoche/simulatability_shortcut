from __future__ import annotations

import torch

from utils.consim import ConSim, PromptSetting, PromptTypes


class ConSimV2(ConSim):
    """ConSim prompt variant framed as simulating another classifier.

    This class intentionally inherits ConSim's validation, sample handling, concept
    formatting, and public API. Only the rendered prompt text differs.
    """

    prompt_types: type[PromptTypes] = PromptTypes

    def construct_prompt(  # type: ignore[override]
        self,
        setting: PromptTypes | PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        *,
        concepts_interpretation: dict[int, str],
        global_importances: torch.Tensor,
        local_importances: list[torch.Tensor] | None = None,
        class_ids: list[int] | None = None,
        top_k: int = 5,
        importance_threshold: float = 0.05,
    ) -> tuple[str, list[str], list[str]]:
        setting = ConSimV2._resolve_prompt_setting(setting)

        self._check_input_settings_correspondence(
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances,
            local_importances=local_importances,
            prompt_type=setting,
            nb_learning_samples=nb_learning_samples,
        )

        classes_ids = (
            sorted(int(class_id) for class_id in class_ids)
            if class_ids is not None
            else sorted(corresponding_predictions.unique().tolist())
        )
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}
        global_importances_dict = {class_id: global_importances[class_id] for class_id in classes_ids}

        return ConSimV2._setting_to_prompt(
            setting=setting,
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            classes=classes,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances_dict,
            local_importances=local_importances,
            top_k=top_k,
            importance_threshold=importance_threshold,
        )

    @staticmethod
    def _setting_to_prompt(  # type: ignore[override]  # noqa: PLR0912
        setting: PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        classes: dict[int, str],
        concepts_interpretation: dict[int, str],
        global_importances: dict[int, torch.Tensor],
        local_importances: list[torch.Tensor] | None,
        top_k: int = 5,
        importance_threshold: float = 0.05,
    ) -> tuple[str, list[str], list[str]]:
        _ = corresponding_labels
        system_prompt_parts = []

        task_description_prompt = (
            "You are simulating a text classifier. For the given evaluation sample, "
            "predict the class label this classifier would assign. Your goal is to "
            "reproduce the classifier's output, even when you would otherwise disagree. "
        )
        if setting.concepts_global_importances:
            task_description_prompt += (
                "You are given the most important concepts for each class according to the classifier. "
            )
        if setting.lp_samples:
            if setting.lp_concepts_local_contributions:
                task_description_prompt += (
                    "You are given examples of inputs, the classifier's predictions, "
                    "and the concept contributions behind those predictions. "
                )
            else:
                task_description_prompt += (
                    "You are given examples of inputs paired with the classifier's predictions. "
                )
        if setting.concepts_global_importances or setting.lp_concepts_local_contributions:
            task_description_prompt += " For each concept, the importances are 'Very opposed', 'Opposed', 'Supportive', or 'Highly supportive'. It means that a opposed concept is present in the text, the corresponding class is improbable. In the other hand, when a supportive concept is present in the text, the corresponding class is more likely."
        task_description_prompt += "Output only the class name the classifier would predict, with no other text."
        system_prompt_parts.append(task_description_prompt)

        if setting.anonymize_classes:
            classes = {i: f"Class_{i}" for i in classes.keys()}
        system_prompt_parts.append(f"The classes are: [{', '.join(list(classes.values()))}]")

        if setting.concepts_global_importances:
            classes_concepts_prompt = (
                "The most important concepts and their importance for each class are:\n"
                + "\n".join(
                    [
                        "\t{}: {}".format(
                            class_name,
                            ConSimV2._concepts_to_string(
                                global_importances[class_index],
                                concepts_interpretation,
                                top_k=top_k,
                                threshold=importance_threshold,
                            ),
                        )
                        for class_index, class_name in classes.items()
                    ]
                )
            )
            system_prompt_parts.append(classes_concepts_prompt)

        if setting.lp_samples:
            learning_phase_blocks = []
            for i in range(nb_learning_samples):
                block = [
                    f"Sample_{i}:",
                    f"\tText: {interesting_samples[i]}",
                    f"\tModel's prediction: {classes[int(corresponding_predictions[i])]}",
                ]

                if setting.lp_concepts_local_contributions:
                    pred = int(corresponding_predictions[i].item())
                    str_importances = ConSimV2._concepts_to_string(
                        local_importances[i][pred],  # type: ignore[index]
                        concepts_interpretation,
                        top_k=top_k,
                        threshold=importance_threshold,
                    )
                    block.append(f"\tConcepts contributions: {str_importances}")

                learning_phase_blocks.append("\n".join(block))
            system_prompt_parts.append("\n".join(learning_phase_blocks))

        system_prompt = "\n\n".join(system_prompt_parts)

        user_prompts = [
            "\n".join(
                [
                    "Evaluation sample:",
                    f"\tText: {interesting_samples[i]}",
                    "\tModel's prediction: ",
                ]
            )
            for i in range(nb_learning_samples, len(interesting_samples))
        ]

        literal_model_predictions = [
            classes[int(corresponding_predictions[i])] for i in range(nb_learning_samples, len(interesting_samples))
        ]

        return system_prompt, user_prompts, literal_model_predictions
