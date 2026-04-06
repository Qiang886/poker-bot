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


def calculate_river_sizing(
    hand_strength: "HandStrength",
    board_texture: "BoardTexture",
    villain_profile,
    pot: float,
    hero_stack: float,
    is_bluff: bool,
    blocker_score: float = 0.0,
) -> SizingProfile:
    """River-specific sizing: more polarized than earlier streets.

    Value bet sizing:
    - Monster (nuts/quads/straight-flush) → overbet 1.25x pot
    - Strong value (flush/straight/full-house/trips) → 66-75% pot
    - Thin value (top pair range) → 33-50% pot
    - vs fish: all value sizing increased by 15%

    Bluff sizing:
    - Standard river bluff → 66-75% pot (need ~40% fold)
    - Small bluff vs nit → 33% pot
    - Overbet bluff with strong blocker → 1.25x pot
    """
    from src.hand_analysis import MadeHandType

    made = hand_strength.made_hand

    # Determine villain type
    player_type = "TAG"
    if villain_profile is not None and villain_profile.stats.hands_played >= 30:
        player_type = villain_profile.classify()

    if is_bluff:
        if player_type == "nit":
            base = SMALL  # 33% pot – nit folds easily
            reason = "river bluff vs nit: small sizing enough"
        elif blocker_score >= 0.6:
            base = OVERBET  # strong blocker → overbet bluff
            reason = f"river overbet bluff: strong blocker ({blocker_score:.2f})"
        else:
            base = LARGE  # standard river bluff 66-75%
            reason = "river bluff: standard sizing"
    else:
        # Value bet sizing
        if made >= MadeHandType.STRAIGHT_FLUSH:
            base = OVERBET
            reason = "river value: nut hand overbet"
        elif made >= MadeHandType.FULL_HOUSE:
            base = OVERBET
            reason = "river value: monster – overbet"
        elif made >= MadeHandType.TRIPS_SET:
            base = LARGE
            reason = "river value: strong hand 66-75%"
        elif made >= MadeHandType.TOP_TWO_PAIR:
            base = LARGE
            reason = "river value: top two pair 66%"
        elif made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            base = MEDIUM + 0.10  # ~60%
            reason = "river value: TPTK 55-66%"
        elif made >= MadeHandType.TOP_PAIR_GOOD_KICKER:
            if player_type == "fish":
                base = MEDIUM  # 50%
                reason = "river thin value vs fish: TPGK 50%"
            elif player_type == "nit":
                base = SMALL  # nit only calls with better; value bet small or check
                reason = "river check/thin-value vs nit: small"
            else:
                base = SMALL  # 33% thin value
                reason = "river thin value: TPGK 33%"
        elif made >= MadeHandType.MIDDLE_PAIR:
            base = SMALL  # thin value only vs fish
            reason = "river thin value: middle pair small"
        elif made in (MadeHandType.OVERPAIR_BIG, MadeHandType.OVERPAIR_SMALL):
            base = MEDIUM  # 50%
            reason = "river value: overpair 50%"
        else:
            base = SMALL
            reason = "river thin value: marginal hand"

        # vs fish: increase all value sizing
        if player_type == "fish":
            base = min(OVERBET, base * 1.15)
            reason += " (+15% vs fish)"

    # Stack-off protection
    actual_amount = pot * base
    if hero_stack > 0 and actual_amount >= hero_stack * 0.90:
        return SizingProfile(
            size_type="all-in",
            fraction_of_pot=hero_stack / pot if pot > 0 else 1.0,
            reasoning="river: stack commitment – all-in",
        )

    return SizingProfile(
        size_type=_size_label(base),
        fraction_of_pot=base,
        reasoning=reason,
    )
