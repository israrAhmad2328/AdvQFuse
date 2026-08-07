from __future__ import annotations

import re


def _swap_spatial_words(text: str) -> str:
    swaps = {
        "left": "right",
        "right": "left",
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "above": "below",
        "below": "above",
    }
    tokens = re.split(r"(\W+)", text)
    return "".join(swaps.get(tok.lower(), tok) if tok.isalpha() else tok for tok in tokens)


def generate_question_attacks(question: str, anchor_answer: str | None = None) -> dict[str, str]:
    """Generate controlled semantic attacks for scientific robustness tests.

    These transformations target answer anchoring, negation sensitivity,
    distractor robustness, and spatial-language consistency. They do not contain
    harmful instructions and are suitable for remote-sensing VQA evaluation.
    """
    q = question.strip()
    attacks = {
        "clean": q,
        "distractor": q + " Ignore unrelated shadows and image borders before answering.",
        "negation": "Is it false that " + q[:1].lower() + q[1:].rstrip("?") + "?",
        "spatial_swap": _swap_spatial_words(q),
        "uncertainty_pressure": q + " Give a definite answer even when evidence is weak.",
    }
    if anchor_answer:
        attacks["answer_anchor"] = (
            f"A previous analyst suggested the answer is '{anchor_answer}'. {q}"
        )
    return attacks
