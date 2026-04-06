"""Bet sizing logic for value and bluff bets."""

from dataclasses import dataclass

from src.hand_analysis import HandStrength, MadeHandType
from src.board_analysis import BoardTexture, RangeAdvantage, NutAdvantage

SMALL = 0.33
MEDIUM = 0.50
LARGE = 0.66
OVERBET = 1.25


@dataclass
class SizingProfile:
    size_type: str
    fraction_of_pot: float
    reasoning: str


def _size_label(fraction: float) -> str:
    if fraction <= SMALL + 0.05:
        return "small"
    if fraction <= MEDIUM + 0.05:
        return "medium"
    if fraction <= LARGE + 0.05:
        return "large"
    if fraction <= OVERBET + 0.10:
        return "overbet"
    return "all-in"


def calculate_sizing(
    hand_strength: HandStrength,
    board_texture: BoardTexture,
    range_advantage: RangeAdvantage,
    street: str,
    spr: float,
    pot: float,
    stack: float,
    is_value: bool,
    is_multiway: bool,
) -> SizingProfile:
    """Determine bet sizing fraction and reasoning."""
    base = MEDIUM
    reasons: list[str] = []

    # Multiway: always use smaller sizes
    if is_multiway:
        base = SMALL
        reasons.append("multiway pot")

    # Strong polar hands size up
    if hand_strength.made_hand >= MadeHandType.STRAIGHT_NON_NUT:
        if not is_multiway:
            base = max(base, LARGE)
            reasons.append("polar value hand")

    if hand_strength.made_hand in (MadeHandType.FLUSH_NUT, MadeHandType.STRAIGHT_FLUSH):
        if not is_multiway:
            base = max(base, OVERBET)
            reasons.append("nut hand – overbet for value")

    # Wet board requires protection
    if board_texture.wetness >= 7 and is_value and not is_multiway:
        base = max(base, LARGE)
        reasons.append("wet board – protection bet")

    # Nut advantage: can use larger size
    if range_advantage.nut_advantage == NutAdvantage.HERO and not is_multiway:
        base = max(base, LARGE)
        reasons.append("nut advantage")

    # Low SPR: go big or all-in
    if spr < 1.5:
        base = OVERBET
        reasons.append("low SPR – commit")
    elif spr < 3:
        base = max(base, LARGE)
        reasons.append("low-medium SPR")

    # River polarisation
    if street == "river" and not is_multiway:
        base = max(base, LARGE)
        reasons.append("river polarisation")

    # Bluffs mirror value for balance; adjust down slightly OOP
    if not is_value:
        base = min(base, LARGE)   # cap bluffs at large
        if not reasons:
            reasons.append("balanced bluff sizing")

    # Ensure we don't over-commit beyond stack
    actual_amount = pot * base
    if stack > 0 and actual_amount >= stack * 0.90:
        return SizingProfile(
            size_type="all-in",
            fraction_of_pot=stack / pot if pot > 0 else 1.0,
            reasoning="stack commitment – all-in",
        )

    reason_str = "; ".join(reasons) if reasons else "standard sizing"
    return SizingProfile(
        size_type=_size_label(base),
        fraction_of_pot=base,
        reasoning=reason_str,
    )
