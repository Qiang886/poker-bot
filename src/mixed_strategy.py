"""GTO-inspired mixed strategy using EV-based softmax selection."""

import math
import random
from typing import Dict, Optional, Tuple


def mixed_decision(
    ev_bet: Optional[float] = None,
    ev_check: Optional[float] = None,
    ev_call: Optional[float] = None,
    ev_fold: Optional[float] = None,
    ev_raise: Optional[float] = None,
    temperature: float = 1.0,
) -> Tuple[str, float]:
    """Select an action using EV-based probabilistic mixing.

    When the EV gap between available actions is large the best action is
    chosen deterministically (pure strategy).  When the gap is small a
    softmax distribution is computed so the bot is not fully predictable.

    Parameters
    ----------
    ev_bet, ev_check, ev_call, ev_fold, ev_raise:
        Expected value for each candidate action.  Pass ``None`` for actions
        that are not available in the current decision point.
    temperature:
        Controls randomness.
        - 0   → always pick the highest-EV action (pure strategy)
        - 1   → standard softmax mixing
        - >1  → more random

    Returns
    -------
    (chosen_action, confidence)
        ``confidence`` is the probability assigned to the chosen action.
    """
    actions: Dict[str, float] = {}
    if ev_bet is not None:
        actions["bet"] = ev_bet
    if ev_check is not None:
        actions["check"] = ev_check
    if ev_call is not None:
        actions["call"] = ev_call
    if ev_fold is not None:
        actions["fold"] = ev_fold
    if ev_raise is not None:
        actions["raise"] = ev_raise

    if not actions:
        return "check", 0.5

    max_ev = max(actions.values())
    min_ev = min(actions.values())
    ev_gap = max_ev - min_ev

    # Large gap or pure-strategy mode → deterministic choice
    if ev_gap >= 0.15 or temperature == 0:
        best = max(actions, key=lambda a: actions[a])
        return best, 0.85

    # Small gap → softmax mixing
    # Standard softmax: exp(ev / temperature). Higher temperature → more uniform.
    scale = max(ev_gap, 0.01)
    scaled: Dict[str, float] = {
        a: math.exp((ev - max_ev) / (temperature * scale)) for a, ev in actions.items()
    }
    total = sum(scaled.values())
    probs: Dict[str, float] = {a: v / total for a, v in scaled.items()}

    r = random.random()
    cumulative = 0.0
    for action, prob in probs.items():
        cumulative += prob
        if r <= cumulative:
            return action, prob

    # Fallback (floating-point edge case)
    best_action = max(actions, key=lambda a: actions[a])
    return best_action, probs.get(best_action, 0.7)
