"""Multiway pot adjustments."""

from dataclasses import dataclass

from src.hand_analysis import HandStrength, MadeHandType, DrawType
from src.board_analysis import BoardTexture


@dataclass
class MultiwayAdjustment:
    bet_frequency_multiplier: float   # multiply base bet frequency
    value_threshold_adjustment: int   # add to MadeHandType threshold
    bluff_allowed: bool
    draw_continue_threshold: DrawType  # minimum draw quality to continue
    sizing_multiplier: float           # multiply bet sizing (1.0 = no change)


def adjust_for_multiway(
    hand_strength: HandStrength,
    num_players: int,
    board_texture: BoardTexture,
) -> MultiwayAdjustment:
    """Return adjustments for multiway pots.

    3-player pot:
    - value_threshold: TOP_PAIR_TOP_KICKER+ (no thin value betting)
    - bluff frequency: -60%
    - draw: combo draws can semi-bluff; single draws cannot
    - sizing: +10-15%

    4+ player pot:
    - value_threshold: TOP_TWO_PAIR+
    - bluff: almost never (-80%+)
    - sizing: +20%+
    - draw: only nut draws continue
    """
    if num_players <= 2:
        return MultiwayAdjustment(
            bet_frequency_multiplier=1.0,
            value_threshold_adjustment=0,
            bluff_allowed=True,
            draw_continue_threshold=DrawType.GUTSHOT,
            sizing_multiplier=1.0,
        )

    draw = hand_strength.draw

    if num_players == 3:
        # 3-way: tighten value range to TPTK+, reduce bluffs, increase sizing
        freq_mult = 0.60
        threshold_adj = int(MadeHandType.TOP_PAIR_TOP_KICKER) - int(MadeHandType.MIDDLE_PAIR)
        # Combo draws can still semi-bluff 3-way
        bluff_ok = draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW)
        draw_threshold = DrawType.COMBO_DRAW
        sizing_mult = 1.12

        if board_texture.wetness >= 7:
            freq_mult = max(0.45, freq_mult - 0.10)
            threshold_adj += 1

    else:
        # 4+ players: very tight, near-zero bluffing
        extra_players = num_players - 3
        freq_mult = max(0.20, 0.40 - extra_players * 0.10)
        threshold_adj = int(MadeHandType.TOP_TWO_PAIR) - int(MadeHandType.MIDDLE_PAIR)
        bluff_ok = draw == DrawType.COMBO_DRAW_NUT
        draw_threshold = DrawType.FLUSH_DRAW_NUT
        sizing_mult = 1.25

        if board_texture.wetness >= 7:
            freq_mult = max(0.15, freq_mult - 0.10)
            threshold_adj += 1

    return MultiwayAdjustment(
        bet_frequency_multiplier=freq_mult,
        value_threshold_adjustment=threshold_adj,
        bluff_allowed=bluff_ok,
        draw_continue_threshold=draw_threshold,
        sizing_multiplier=sizing_mult,
    )
