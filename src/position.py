"""Seat positions for 6-max No-Limit Hold'em."""

from enum import IntEnum
from typing import List


class Position(IntEnum):
    UTG = 0
    HJ = 1
    CO = 2
    BTN = 3
    SB = 4
    BB = 5


POSITION_NAMES = {
    Position.UTG: "UTG",
    Position.HJ: "HJ",
    Position.CO: "CO",
    Position.BTN: "BTN",
    Position.SB: "SB",
    Position.BB: "BB",
}


def get_position_name(pos: Position) -> str:
    return POSITION_NAMES[pos]


def is_in_position(hero_pos: Position, villain_pos: Position) -> bool:
    """Return True if hero acts after villain postflop (hero is in position)."""
    # Postflop order: SB acts first, then BB, UTG, HJ, CO, BTN acts last
    postflop_order = [
        Position.SB, Position.BB, Position.UTG,
        Position.HJ, Position.CO, Position.BTN,
    ]
    return postflop_order.index(hero_pos) > postflop_order.index(villain_pos)


def get_rfi_position_order() -> List[Position]:
    """Preflop action order for RFI (first raise)."""
    return [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]
