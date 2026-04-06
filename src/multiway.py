"""Multiway pot adjustments."""

from dataclasses import dataclass

from src.hand_analysis import HandStrength, MadeHandType, DrawType
from src.board_analysis import BoardTexture


@dataclass
class MultiwayAdjustment:
    bet_frequency_multiplier: float   # multiply base bet frequency
    value_threshold_adjustment: int   # add to MadeHandType threshold
    bluff_allowed: bool


def adjust_for_multiway(
    hand_strength: HandStrength,
    num_players: int,
    board_texture: BoardTexture,
) -> MultiwayAdjustment:
    """Return adjustments to apply to normal heads-up strategy for multiway pots.

    Key principles:
    - Almost no bluffing (only nut combo draws)
    - Tighten value range significantly
    - Reduce bet frequency
    """
    if num_players <= 2:
        # Heads-up: no adjustments needed
        return MultiwayAdjustment(
            bet_frequency_multiplier=1.0,
            value_threshold_adjustment=0,
            bluff_allowed=True,
        )

    extra_players = num_players - 2   # 1 extra = 3-way, 2 extra = 4-way, etc.

    # Bet frequency reduces with each additional player
    freq_mult = max(0.20, 1.0 - extra_players * 0.25)

    # Tighten value threshold: require stronger hands
    threshold_adj = extra_players * 2   # e.g., 3-way: need TOP_PAIR_GOOD_KICKER+2 = TOP_TWO_PAIR

    # Bluffing only allowed with nut combo draws
    draw = hand_strength.draw
    bluff_ok = draw in (DrawType.COMBO_DRAW_NUT, DrawType.FLUSH_DRAW_NUT)

    # Wet boards: tighten further
    if board_texture.wetness >= 7:
        freq_mult = max(0.15, freq_mult - 0.10)
        threshold_adj += 1

    return MultiwayAdjustment(
        bet_frequency_multiplier=freq_mult,
        value_threshold_adjustment=threshold_adj,
        bluff_allowed=bluff_ok,
    )
