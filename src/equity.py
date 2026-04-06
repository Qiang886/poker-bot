"""Monte Carlo equity calculator."""

import random
from typing import List, Tuple

from src.card import Card, FULL_DECK
from src.evaluator import evaluate_7
from src.weighted_range import ComboWeight


def _sample_villain(villain_range: List[ComboWeight]) -> Tuple[Card, Card]:
    """Weighted random sample of a villain combo."""
    total = sum(cw.weight for cw in villain_range)
    r = random.random() * total
    cumulative = 0.0
    for cw in villain_range:
        cumulative += cw.weight
        if r <= cumulative:
            return cw.combo
    return villain_range[-1].combo


def calculate_equity(
    hero_hand: Tuple[Card, Card],
    villain_range: List[ComboWeight],
    board: List[Card],
    num_simulations: int = 1000,
) -> float:
    """Monte Carlo equity for hero_hand vs villain_range on board.

    Returns hero's equity as a fraction [0.0, 1.0].
    """
    if not villain_range:
        return 0.5

    dead_base = set(hero_hand) | set(board)
    available_base = [c for c in FULL_DECK if c not in dead_base]

    wins = 0.0
    total = 0

    for _ in range(num_simulations):
        villain_combo = _sample_villain(villain_range)
        if villain_combo[0] in dead_base or villain_combo[1] in dead_base:
            continue

        villain_set = set(villain_combo)
        available = [c for c in available_base if c not in villain_set]
        needed = 5 - len(board)
        if needed > len(available):
            continue

        runout = random.sample(available, needed)
        full_board = list(board) + runout

        hero_eval = evaluate_7(list(hero_hand) + full_board)
        villain_eval = evaluate_7(list(villain_combo) + full_board)

        if hero_eval > villain_eval:
            wins += 1.0
        elif hero_eval == villain_eval:
            wins += 0.5
        total += 1

    if total == 0:
        return 0.5
    return wins / total


def calculate_range_equity(
    hero_range: List[ComboWeight],
    villain_range: List[ComboWeight],
    board: List[Card],
    num_simulations: int = 500,
) -> float:
    """Range vs range equity via Monte Carlo.

    Returns hero range equity as a fraction [0.0, 1.0].
    """
    if not hero_range or not villain_range:
        return 0.5

    board_set = set(board)
    wins = 0.0
    total = 0

    for _ in range(num_simulations):
        hero_combo = _sample_villain(hero_range)
        villain_combo = _sample_villain(villain_range)

        used = board_set | set(hero_combo) | set(villain_combo)
        if len(used) < len(board) + 4:
            # Overlap between combos or board
            continue

        available = [c for c in FULL_DECK if c not in used]
        needed = 5 - len(board)
        if needed > len(available):
            continue

        runout = random.sample(available, needed)
        full_board = list(board) + runout

        hero_eval = evaluate_7(list(hero_combo) + full_board)
        villain_eval = evaluate_7(list(villain_combo) + full_board)

        if hero_eval > villain_eval:
            wins += 1.0
        elif hero_eval == villain_eval:
            wins += 0.5
        total += 1

    if total == 0:
        return 0.5
    return wins / total
