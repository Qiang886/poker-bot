"""Dynamic equity bucket adjustment based on villain type, board texture, and street."""

from src.hand_analysis import HandStrength, MadeHandType, DrawType
from src.board_analysis import BoardTexture


def adjust_equity_bucket(
    hand_strength: HandStrength,
    board_texture: BoardTexture,
    villain_type: str,
    street: str,
    turn_is_blank: bool = True,
    river_is_blank: bool = True,
) -> float:
    """Dynamically adjust equity bucket based on context.

    Args:
        hand_strength: Classified hand strength (from classify_hand).
        board_texture: Analyzed board texture (from analyze_board).
        villain_type: One of "fish", "nit", "LAG", "TAG", "unknown".
        street: Current street: "flop", "turn", or "river".
        turn_is_blank: True if the turn card was a blank (no dangerous tag).
        river_is_blank: True if the river card was a blank.

    Returns:
        Adjusted equity bucket clamped to [0.01, 0.99].
    """
    base = hand_strength.equity_bucket
    made = hand_strength.made_hand

    # ------------------------------------------------------------------
    # Villain-type adjustment
    # ------------------------------------------------------------------
    type_adj = 0.0
    if villain_type == "fish":
        # Fish range is wide; our hand is relatively stronger
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
        # Nit only plays premiums; our hand is relatively weaker
        if made >= MadeHandType.TRIPS_SET:
            type_adj = -0.03
        elif made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            type_adj = -0.12
        elif made >= MadeHandType.MIDDLE_PAIR:
            type_adj = -0.12
        else:
            type_adj = -0.10
    elif villain_type == "LAG":
        # LAG plays wide; mild positive adjustment for made hands
        if made >= MadeHandType.TOP_PAIR_WEAK_KICKER:
            type_adj = 0.06
        elif made >= MadeHandType.MIDDLE_PAIR:
            type_adj = 0.05
        else:
            type_adj = 0.03

    # ------------------------------------------------------------------
    # Board texture adjustment
    # ------------------------------------------------------------------
    texture_adj = 0.0
    if board_texture.wetness >= 7:
        texture_adj = -0.06
    elif board_texture.wetness <= 3:
        texture_adj = 0.04

    # Flush possible but hero has no flush draw → hero is behind on this axis
    if (
        board_texture.flush_possible
        and hand_strength.draw == DrawType.NONE
        and made < MadeHandType.FLUSH_LOW
    ):
        texture_adj -= 0.05

    # Paired board: risk of full house / quads if hero doesn't have trips/set
    if board_texture.paired and made < MadeHandType.TRIPS_SET:
        texture_adj -= 0.04

    # ------------------------------------------------------------------
    # Street adjustment
    # ------------------------------------------------------------------
    street_adj = 0.0
    if street == "turn":
        street_adj = 0.03 if turn_is_blank else -0.08
    elif street == "river":
        street_adj = 0.05 if river_is_blank else -0.12

    adjusted = base + type_adj + texture_adj + street_adj
    return max(0.01, min(0.99, adjusted))
