"""Prompt construction and generated output parsing for RLLaMA."""

import re
from typing import Dict, List, Optional


def build_prompt(
    text: str, retrieved: List[Dict[str, object]], include_labels: bool = True
) -> str:
    """Builds a retrieval-augmented classification prompt with top-k reranked examples."""
    context_lines = []
    for i, item in enumerate(retrieved, start=1):
        lab = "Fake" if int(item["label"]) == 0 else "Real"
        if include_labels:
            context_lines.append(
                f"Example {i} (training label: {lab}): {item['text']}"
            )
        else:
            context_lines.append(f"Related training text {i}: {item['text']}")
    contexts = (
        "\n".join(context_lines)
        if context_lines
        else "No related training examples were retrieved."
    )
    return f"""You are a careful fake-news classification assistant.
Classify the post as Fake or Real based on the post text and the similar labelled training examples.
The retrieved examples are similarity-based training examples, not independently verified evidence.
Do not invent facts. If the text is insufficient, choose the most plausible label but explain uncertainty briefly.

SIMILAR TRAINING EXAMPLES:
{contexts}

POST TO CLASSIFY:
{text}

Return exactly this format:
Label: Fake or Real
Reason: one short sentence
"""


def parse_label(generated: str) -> Optional[int]:
    """Extracts the predicted binary label (0=Fake, 1=Real) from LLM output."""
    matches = re.findall(r"(?im)^\s*Label\s*:\s*(Fake|Real)\b", generated)
    if matches:
        return 0 if matches[-1].lower() == "fake" else 1
    fallback = re.search(r"(?i)\b(Fake|Real)\b", generated)
    if fallback:
        return 0 if fallback.group(1).lower() == "fake" else 1
    return None
