"""Multi-street barrel planning for post-flop play."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.card import Card
from src.hand_analysis import HandStrength, MadeHandType, DrawType
from src.board_analysis import BoardTexture, RangeAdvantage, NutAdvantage


class ValueLine(Enum):
    BET_BET_BET = "bet_bet_bet"
    BET_BET_CHECK = "bet_bet_check"
    BET_CHECK_BET = "bet_check_bet"
    BET_CHECK_CHECK = "bet_check_check"
    CHECK_CALL = "check_call"
    CHECK_RAISE = "check_raise"


class BluffLine(Enum):
    BLUFF_ONE_AND_DONE = "bluff_one_and_done"
    BLUFF_DOUBLE_BARREL = "bluff_double_barrel"
    BLUFF_TRIPLE_BARREL = "bluff_triple_barrel"
    DELAYED_CBET = "delayed_cbet"
    CHECK_RAISE_BLUFF = "check_raise_bluff"


class PassiveLine(Enum):
    CHECK_FOLD = "check_fold"
    GIVE_UP = "give_up"


@dataclass
class BarrelPlan:
    flop_action: str    # "bet" or "check"
    turn_action: str    # "bet", "check", "barrel"
    river_action: str   # "bet", "check", "bluff"
    value_line: Optional[ValueLine] = None
    bluff_line: Optional[BluffLine] = None
    continue_runouts: List[str] = field(default_factory=list)
    give_up_runouts: List[str] = field(default_factory=list)


def create_barrel_plan(
    hand_strength: HandStrength,
    board: BoardTexture,
    range_advantage: RangeAdvantage,
    position: str,           # "ip" or "oop"
    spr: float,
    villain_profile=None,    # Optional[VillainProfile] – avoid circular import
    outs_analysis=None,      # Optional[OutsAnalysis] from outs_calculator
) -> BarrelPlan:
    """Choose a multi-street plan based on hand strength, board, and position."""
    from src.outs_calculator import OutsAnalysis  # local import to avoid circular
    made = hand_strength.made_hand
    draw = hand_strength.draw
    in_position = position.lower() == "ip"
    has_nut_adv = range_advantage.nut_advantage == NutAdvantage.HERO

    # ------------------------------------------------------------------
    # Monster hands: always 3-barrel for value
    # ------------------------------------------------------------------
    if made >= MadeHandType.TRIPS_SET:
        # Slow-play sets on dry boards OOP
        if made == MadeHandType.TRIPS_SET and not board.flush_possible and not board.straight_possible and not in_position:
            return BarrelPlan(
                flop_action="check",
                turn_action="bet",
                river_action="bet",
                value_line=ValueLine.BET_CHECK_BET,
                continue_runouts=["any"],
                give_up_runouts=[],
            )
        return BarrelPlan(
            flop_action="bet",
            turn_action="bet",
            river_action="bet",
            value_line=ValueLine.BET_BET_BET,
            continue_runouts=["any"],
            give_up_runouts=[],
        )

    # ------------------------------------------------------------------
    # Strong value: top pair top kicker / overpairs / two pair / straights / flushes
    # ------------------------------------------------------------------
    if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
        if hand_strength.is_vulnerable and board.wetness >= 6:
            # Protect on wet boards aggressively
            return BarrelPlan(
                flop_action="bet",
                turn_action="bet",
                river_action="check",
                value_line=ValueLine.BET_BET_CHECK,
                continue_runouts=["blank", "low_card"],
                give_up_runouts=["flush_completes", "straight_completes"],
            )
        if made >= MadeHandType.TOP_TWO_PAIR or made >= MadeHandType.STRAIGHT_NON_NUT:
            return BarrelPlan(
                flop_action="bet",
                turn_action="bet",
                river_action="bet",
                value_line=ValueLine.BET_BET_BET,
                continue_runouts=["any"],
                give_up_runouts=[],
            )
        # Top pair good kicker / overpair: bet-bet-check or bet-bet-bet
        if spr > 3:
            return BarrelPlan(
                flop_action="bet",
                turn_action="bet",
                river_action="check",
                value_line=ValueLine.BET_BET_CHECK,
                continue_runouts=["blank"],
                give_up_runouts=["flush_completes", "straight_completes"],
            )
        # Short SPR: go for it
        return BarrelPlan(
            flop_action="bet",
            turn_action="bet",
            river_action="bet",
            value_line=ValueLine.BET_BET_BET,
            continue_runouts=["any"],
            give_up_runouts=[],
        )

    # ------------------------------------------------------------------
    # Medium value: top pair weak kicker, middle pair, overpair small
    # ------------------------------------------------------------------
    if made >= MadeHandType.MIDDLE_PAIR:
        if in_position:
            return BarrelPlan(
                flop_action="bet",
                turn_action="check",
                river_action="check",
                value_line=ValueLine.BET_CHECK_CHECK,
                continue_runouts=["blank"],
                give_up_runouts=["wet_turns"],
            )
        # OOP: check-call
        return BarrelPlan(
            flop_action="check",
            turn_action="check",
            river_action="check",
            value_line=ValueLine.CHECK_CALL,
            continue_runouts=["blank"],
            give_up_runouts=["wet_turns"],
        )

    # ------------------------------------------------------------------
    # Nut draws: semi-bluff aggressively
    # Use outs_analysis if available to calibrate barrel count
    # ------------------------------------------------------------------
    if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT):
        clean_outs = outs_analysis.total_clean if outs_analysis is not None else 9
        if clean_outs >= 12:
            # Combo draw (12+ clean outs) → full triple barrel potential
            return BarrelPlan(
                flop_action="bet",
                turn_action="bet",
                river_action="bluff",
                value_line=None,
                bluff_line=BluffLine.BLUFF_TRIPLE_BARREL,
                continue_runouts=["flush_completes", "straight_completes"],
                give_up_runouts=[],
            )
        if in_position:
            return BarrelPlan(
                flop_action="bet",
                turn_action="bet",
                river_action="bluff",
                value_line=None,
                bluff_line=BluffLine.BLUFF_DOUBLE_BARREL,
                continue_runouts=["flush_completes", "straight_completes"],
                give_up_runouts=["paired_board"],
            )
        # OOP: check-raise as semi-bluff, then barrel
        return BarrelPlan(
            flop_action="check",
            turn_action="bet",
            river_action="bluff",
            value_line=None,
            bluff_line=BluffLine.CHECK_RAISE_BLUFF,
            continue_runouts=["flush_completes"],
            give_up_runouts=["paired_board"],
        )

    # ------------------------------------------------------------------
    # Combo draw (non-nut): semi-bluff but give up if missed
    # Use outs_analysis if available
    # ------------------------------------------------------------------
    if draw in (DrawType.COMBO_DRAW, DrawType.OESD):
        clean_outs = outs_analysis.total_clean if outs_analysis is not None else 8
        if clean_outs >= 8:
            # Strong draw: semi-bluff double barrel
            return BarrelPlan(
                flop_action="bet" if in_position else "check",
                turn_action="bet",
                river_action="check",
                value_line=None,
                bluff_line=BluffLine.BLUFF_DOUBLE_BARREL if in_position else BluffLine.BLUFF_ONE_AND_DONE,
                continue_runouts=["straight_completes"],
                give_up_runouts=["blank"],
            )
        if clean_outs <= 4 and (outs_analysis is None or outs_analysis.total_dirty >= clean_outs):
            # Weak/dirty draw: check-fold
            return BarrelPlan(
                flop_action="check",
                turn_action="check",
                river_action="check",
                value_line=None,
                bluff_line=None,
                continue_runouts=[],
                give_up_runouts=["any"],
            )
        return BarrelPlan(
            flop_action="bet" if in_position else "check",
            turn_action="bet",
            river_action="check",
            value_line=None,
            bluff_line=BluffLine.BLUFF_ONE_AND_DONE if not in_position else BluffLine.DELAYED_CBET,
            continue_runouts=["straight_completes"],
            give_up_runouts=["blank"],
        )

    # ------------------------------------------------------------------
    # Backdoor equity + range advantage: delayed cbet
    # ------------------------------------------------------------------
    if has_nut_adv and made >= MadeHandType.ACE_HIGH:
        return BarrelPlan(
            flop_action="bet",
            turn_action="check",
            river_action="check",
            value_line=None,
            bluff_line=BluffLine.BLUFF_ONE_AND_DONE,
            continue_runouts=[],
            give_up_runouts=["any"],
        )

    # ------------------------------------------------------------------
    # Air / weak hands: check-fold
    # ------------------------------------------------------------------
    return BarrelPlan(
        flop_action="check",
        turn_action="check",
        river_action="check",
        value_line=None,
        bluff_line=None,
        continue_runouts=[],
        give_up_runouts=["any"],
    )


def get_current_action(
    plan: BarrelPlan,
    street: str,
    turn_card: Optional[Card] = None,
    river_card: Optional[Card] = None,
) -> str:
    """Return the recommended action for the current street given the plan."""
    street = street.lower()
    if street == "flop":
        return plan.flop_action
    if street == "turn":
        # Check if turn card hits a give-up runout
        if plan.give_up_runouts and "any" in plan.give_up_runouts:
            return "check"
        return plan.turn_action if plan.turn_action else "check"
    if street == "river":
        if plan.give_up_runouts and "any" in plan.give_up_runouts:
            return "check"
        action = plan.river_action if plan.river_action else "check"
        if action == "bluff" and plan.bluff_line is None:
            # river_action was set to "bluff" but no bluff_line was assigned;
            # fall back to a passive check rather than attempting an unplanned bluff
            return "check"
        return action
    return "check"
