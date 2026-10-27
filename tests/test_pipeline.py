"""Unit tests for RLLaMA data preprocessing, splitting, and prompt/label parsing."""

import unittest
import pandas as pd
from src.data_utils import make_stratified_splits, map_label, prepare_dataframe
from src.prompts import build_prompt, parse_label


class TestRLLaMAPipeline(unittest.TestCase):

    def test_map_label(self):
        self.assertEqual(map_label("Fake"), 0)
        self.assertEqual(map_label("0"), 0)
        self.assertEqual(map_label("Real"), 1)
        self.assertEqual(map_label(1), 1)
        with self.assertRaises(ValueError):
            map_label("unknown_class")

    def test_prepare_and_split_dataframe(self):
        raw = pd.DataFrame({
            "title": [f"Real headline {i}" for i in range(20)]
            + [f"Fake headline {i}" for i in range(20)],
            "real": [1] * 20 + [0] * 20,
        })
        df = prepare_dataframe(raw, text_col="title", label_col="real")
        self.assertEqual(len(df), 40)
        train_df, val_df, test_df = make_stratified_splits(df, seed=42)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 40)

    def test_build_prompt_and_parse_label(self):
        retrieved = [{"text": "Sample headline", "label": 0}]
        prompt = build_prompt("Query headline", retrieved)
        self.assertIn("Example 1 (training label: Fake): Sample headline", prompt)
        self.assertEqual(parse_label("Label: Fake\nReason: Sensational claim."), 0)
        self.assertEqual(parse_label("Label: Real\nReason: Verified report."), 1)
        self.assertIsNone(parse_label("Unable to classify."))


if __name__ == "__main__":
    unittest.main()
