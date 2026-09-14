import unittest

from vlm_model_selection import VLMModelProfile, select_vlm_model


class VLMModelSelectionTests(unittest.TestCase):
    def test_balanced_defaults_to_general_purpose_model(self) -> None:
        self.assertEqual(select_vlm_model(priority="balanced"), "gpt-4o-mini")

    def test_quality_priority_prefers_high_quality_model(self) -> None:
        self.assertEqual(select_vlm_model(priority="quality"), "gpt-4o")

    def test_video_requirement_filters_models(self) -> None:
        self.assertEqual(select_vlm_model(require_video=True), "gpt-4o")

    def test_raises_when_no_model_matches(self) -> None:
        models = (
            VLMModelProfile(
                name="ocr-only",
                quality_score=0.7,
                speed_score=0.8,
                cost_score=0.8,
                supports_ocr=True,
            ),
        )
        with self.assertRaises(ValueError):
            select_vlm_model(require_video=True, available_models=models)


if __name__ == "__main__":
    unittest.main()
