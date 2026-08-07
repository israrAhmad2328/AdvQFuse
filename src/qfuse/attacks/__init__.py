"""Adversarial attacks for remote-sensing vision-language evaluation."""

from .cross_modal import compose_sensor_panel, render_instruction_overlay
from .image import (
    apply_patch,
    fgsm_linf,
    pgd_linf,
    random_patch,
    spsa_patch_attack,
)
from .text import generate_question_attacks

__all__ = [
    "apply_patch",
    "fgsm_linf",
    "pgd_linf",
    "random_patch",
    "spsa_patch_attack",
    "generate_question_attacks",
    "compose_sensor_panel",
    "render_instruction_overlay",
]
