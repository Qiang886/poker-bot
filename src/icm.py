"""ICM (Independent Chip Model) calculations."""

from typing import List, Tuple


def calculate_icm(stacks: List[float], payouts: List[float]) -> List[float]:
    """Standard recursive ICM equity calculation."""
    n = len(stacks)
    total = sum(stacks)
    equities = [0.0] * n

    def recurse(remaining: List[int], chips_left: float, payout_idx: int, prob: float) -> None:
        if payout_idx >= len(payouts) or not remaining:
            return
        for i in remaining:
            p = stacks[i] / chips_left
            equities[i] += prob * p * payouts[payout_idx]
            new_remaining = [j for j in remaining if j != i]
            recurse(new_remaining, chips_left - stacks[i], payout_idx + 1, prob * p)

    recurse(list(range(n)), total, 0, 1.0)
    return equities


def icm_pressure(
    hero_stack: float,
    all_stacks: List[float],
    payouts: List[float],
    hero_index: int,
) -> float:
    """Return ICM pressure 0-1; higher = more conservative play warranted."""
    if not payouts or sum(all_stacks) == 0:
        return 0.0
    equities = calculate_icm(all_stacks, payouts)
    total_chips = sum(all_stacks)
    chip_ev = hero_stack / total_chips * sum(payouts)
    icm_ev = equities[hero_index]
    if chip_ev == 0:
        return 0.0
    # Pressure = how much ICM undervalues chips relative to chip-EV
    pressure = max(0.0, min(1.0, 1.0 - (icm_ev / chip_ev)))
    return pressure


def adjust_strategy_for_icm(
    action: str,
    amount: float,
    pot: float,
    icm_pressure_val: float,
    stack_to_blind: float,
) -> Tuple[str, float]:
    """Adjust a poker decision based on ICM pressure."""
    if icm_pressure_val < 0.10:
        return action, amount

    # High pressure: fold marginal spots
    if icm_pressure_val > 0.50:
        if action in ("call", "raise") and amount > pot * 0.5:
            return "fold", 0.0
        if action == "bet" and amount > pot * 0.6:
            new_amount = pot * 0.4
            return "bet", new_amount

    # Bubble: short stacks shove, medium stacks fold more
    if stack_to_blind < 10 and action in ("raise", "call"):
        return "all-in", amount

    if icm_pressure_val > 0.30 and action == "call":
        if amount > pot * 0.4:
            return "fold", 0.0

    return action, amount
