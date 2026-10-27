"""Experiment and model configuration for RLLaMA Fake News Detection.

All file paths, column names, and model identifiers are configurable via
environment variables or dataclass fields.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, Optional


DEFAULT_LABEL_MAP: Dict[str, int] = {
    "fake": 0,
    "false": 0,
    "rumor": 0,
    "rumour": 0,
    "0": 0,
    "real": 1,
    "true": 1,
    "non-rumor": 1,
    "non_rumor": 1,
    "1": 1,
}


@dataclass
class ExperimentConfig:
    """Configuration parameters for data splitting, retrieval, QLoRA, and SFT."""

    seed: int = int(os.getenv("RLLAMA_SEED", "42"))
    csv_path: str = os.getenv("RLLAMA_CSV_PATH", "data/FakeNewsNet.csv")
    text_col: str = os.getenv("RLLAMA_TEXT_COL", "title")
    label_col: str = os.getenv("RLLAMA_LABEL_COL", "real")
    base_model: str = os.getenv(
        "RLLAMA_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    fallback_model: str = "Qwen/Qwen2.5-3B-Instruct"
    embed_model_name: str = os.getenv(
        "RLLAMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    reranker_name: str = os.getenv(
        "RLLAMA_RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    top_candidates: int = 20
    top_k: int = 4
    max_length: int = 1024
    epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_train_rows: Optional[int] = None
    max_test_rows: Optional[int] = None
    output_dir: str = os.getenv("RLLAMA_OUTPUT_DIR", "outputs/rankrag_fake_news")
    label_map: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LABEL_MAP)
    )
