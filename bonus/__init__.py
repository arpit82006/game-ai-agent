"""
Package: bonus
Purpose:
    Special / Bonus Level management, progressive reveal planning,
    structural verification, and reveal loop execution for Ball Sort AI.
"""

from bonus.models import RevealMove, RevealStepResult, BonusRevealReport, RevealStepContext, RevealLifecycleState, DestinationMode
from bonus.detector import is_bonus_level
from bonus.planner import get_legal_reveal_moves, select_best_reveal_move
from bonus.verifier import verify_reveal_transition
from bonus.controller import run_bonus_reveal_loop, run_bonus_reveal_single_step, create_reveal_context

__all__ = [
    "RevealMove",
    "RevealStepResult",
    "BonusRevealReport",
    "RevealStepContext",
    "RevealLifecycleState",
    "DestinationMode",
    "is_bonus_level",
    "get_legal_reveal_moves",
    "select_best_reveal_move",
    "verify_reveal_transition",
    "run_bonus_reveal_loop",
    "run_bonus_reveal_single_step",
    "create_reveal_context"
]
