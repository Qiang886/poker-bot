"""Dynamic equity adjustment based on opponent type, board texture, and street."""

from src.hand_analysis import MadeHandType, DrawType, HandStrength
from src.board_analysis import BoardTexture


def adjust_equity_bucket(
    hand_strength: HandStrength,
    board_texture: BoardTexture,
    villain_type: str,       # "fish", "nit", "LAG", "TAG", "unknown"
    street: str,             # "flop", "turn", "river"
    turn_is_blank: bool = True,
    river_is_blank: bool = True,
) -> float:
    """
    Dynamically adjust equity bucket based on opponent type, board texture, and street.

    Returns an adjusted equity value in [0.0, 1.0].

    vs fish: equity increases (fish has wide, weak range)
    vs nit:  equity decreases (nit has strong, narrow range)
    vs LAG:  small positive adjustment (showdown value against bluffs)
    vs TAG:  no adjustment (baseline)

    Wet board lowers made-hand equity; dry board raises it.
    Board pairing without trips/full house lowers equity.
    Street adjustments account for draw completion probability.
    """
    base = hand_strength.equity_bucket
    made = hand_strength.made_hand

    # Draw hands: do not adjust (draw equity is governed by outs, not bucket)
    if hand_strength.draw not in (DrawType.NONE,):
        if made < MadeHandType.BOTTOM_PAIR:
            return base

    # === Opponent type adjustment ===
    type_adj = 0.0
    if villain_type == "fish":
        if made >= MadeHandType.TRIPS_SET:
            type_adj = 0.03
        elif made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            type_adj = 0.12
        elif made >= MadeHandType.MIDDLE_PAIR:
            type_adj = 0.10
        elif made >= MadeHandType.BOTTOM_PAIR:
            type_adj = 0.08
        elif made == MadeHandType.ACE_HIGH:
            type_adj = 0.10
        else:
            type_adj = 0.05
    elif villain_type == "nit":
        if made >= MadeHandType.TRIPS_SET:
            type_adj = -0.03
        elif made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            type_adj = -0.12
        elif made >= MadeHandType.MIDDLE_PAIR:
            type_adj = -0.12
        elif made >= MadeHandType.BOTTOM_PAIR:
            type_adj = -0.10
        elif made == MadeHandType.ACE_HIGH:
            type_adj = -0.10
        else:
            type_adj = -0.05
    elif villain_type == "LAG":
        if made >= MadeHandType.TRIPS_SET:
            type_adj = 0.02
        elif made >= MadeHandType.TOP_PAIR_WEAK_KICKER:
            type_adj = 0.06
        elif made >= MadeHandType.MIDDLE_PAIR:
            type_adj = 0.05
        else:
            type_adj = 0.03
    # TAG and unknown: type_adj = 0.0

    # === Board texture adjustment ===
    texture_adj = 0.0
    if board_texture.wetness >= 7:
        texture_adj -= 0.06
    elif board_texture.wetness <= 3:
        texture_adj += 0.04

    # Flush-possible board without hero having a flush
    if board_texture.flush_possible and hand_strength.draw == DrawType.NONE:
        if made < MadeHandType.FLUSH_LOW:
            texture_adj -= 0.05

    # Paired board without trips/full house
    if board_texture.paired and made < MadeHandType.TRIPS_SET:
        texture_adj -= 0.04

    # === Street adjustment ===
    street_adj = 0.0
    if street == "turn":
        if turn_is_blank:
            street_adj = 0.03   # one draw card consumed; made hand safer
        else:
            street_adj = -0.08  # scary/draw-completing turn hurts
    elif street == "river":
        if river_is_blank:
            street_adj = 0.05   # all draws missed; made hand is final value
        else:
            street_adj = -0.12  # draw-completing river hurts significantly

    adjusted = base + type_adj + texture_adj + street_adj
    return max(0.01, min(0.99, adjusted))
