"""Multi-street barrel planning for post-flop play."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.card import Card, Rank
from src.hand_analysis import HandStrength, MadeHandType, DrawType
from src.board_analysis import BoardTexture, RangeAdvantage, NutAdvantage


# Default outs assumptions when OutsAnalysis is not provided
_DEFAULT_NUT_FLUSH_CLEAN_OUTS = 9   # typical flush draw clean outs
_DEFAULT_OESD_CLEAN_OUTS = 8        # typical OESD clean outs


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
        # OOP set on dry board: check-raise trap to build pot
        if made == MadeHandType.TRIPS_SET and not board.flush_possible and not board.straight_possible and not in_position:
            return BarrelPlan(
                flop_action="check",
                turn_action="bet",
                river_action="bet",
                value_line=ValueLine.CHECK_RAISE,
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
        clean_outs = outs_analysis.total_clean if outs_analysis is not None else _DEFAULT_NUT_FLUSH_CLEAN_OUTS
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
        clean_outs = outs_analysis.total_clean if outs_analysis is not None else _DEFAULT_OESD_CLEAN_OUTS
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


def analyze_runout(existing_board: List[Card], new_card: Card) -> List[str]:
    """Analyze a new turn or river card and return descriptive tags.

    Tags returned (may include multiple):
    - "flush_completes": new card brings 3+ cards of the same suit for the first time
    - "monotone_4": board now has 4+ cards of the same suit
    - "straight_completes": new card creates 3+ consecutive rank values (new milestone)
    - "overcard": new card ranks higher than every card on the existing board
    - "paired_board": new card pairs the board for the first time
    - "scary_card": new card is an Ace or King arriving on a medium/low board
    - "low_card": new card is a 2-6
    - "blank": no dangerous tags (safe runout)
    """
    tags: List[str] = []
    full_board = existing_board + [new_card]

    # Suit analysis
    suits_full = Counter(c.suit for c in full_board)
    suits_old = Counter(c.suit for c in existing_board)
    max_suit_full = max(suits_full.values()) if suits_full else 0
    max_suit_old = max(suits_old.values()) if suits_old else 0

    if max_suit_full >= 3 and max_suit_old < 3:
        tags.append("flush_completes")
    if max_suit_full >= 4:
        tags.append("monotone_4")

    # Pairing detection
    ranks_full = [c.rank for c in full_board]
    ranks_old = [c.rank for c in existing_board]
    rank_counts_full = Counter(ranks_full)
    rank_counts_old = Counter(ranks_old)
    old_max_count = max(rank_counts_old.values()) if rank_counts_old else 0
    new_max_count = max(rank_counts_full.values()) if rank_counts_full else 0
    if new_max_count >= 2 and old_max_count < 2:
        tags.append("paired_board")

    # Overcard detection
    if existing_board:
        board_max_rank = max(c.rank for c in existing_board)
        if new_card.rank > board_max_rank:
            tags.append("overcard")

    # Scary card: Ace or King arriving on a medium/low board
    if existing_board:
        board_max_val = max(c.rank for c in existing_board)
        if new_card.rank in (Rank.ACE, Rank.KING) and board_max_val <= Rank.QUEEN:
            tags.append("scary_card")

    # Straight connectivity: count max consecutive run of distinct rank values
    def _max_consecutive(cards: List[Card]) -> int:
        vals = sorted(set(c.rank for c in cards))
        if not vals:
            return 0
        best = cur = 1
        for i in range(1, len(vals)):
            if vals[i].value == vals[i - 1].value + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    consec_old = _max_consecutive(existing_board)
    consec_new = _max_consecutive(full_board)
    if consec_new >= 3 and consec_new > consec_old:
        tags.append("straight_completes")

    # Low card (2-6)
    if new_card.rank <= Rank.SIX:
        tags.append("low_card")

    # Blank: no dangerous tags
    danger_tags = {
        "flush_completes", "straight_completes", "overcard",
        "paired_board", "scary_card", "monotone_4",
    }
    if not any(t in danger_tags for t in tags):
        tags.append("blank")

    return tags


def get_current_action(
    plan: BarrelPlan,
    street: str,
    turn_card: Optional[Card] = None,
    river_card: Optional[Card] = None,
    board: Optional[List[Card]] = None,
) -> str:
    """Return the recommended action for the current street given the plan.

    When ``board`` and the street card (``turn_card`` / ``river_card``) are
    provided, ``analyze_runout`` is called to decide whether to deviate from
    the plan based on the texture of the new card.
    """
    street = street.lower()

    if street == "flop":
        return plan.flop_action

    if street == "turn":
        # Global give-up override
        if plan.give_up_runouts and "any" in plan.give_up_runouts:
            return "check"

        if turn_card is not None and board is not None:
            # board[:3] is the flop; turn_card is the new card
            flop = board[:3]
            runout_tags = analyze_runout(flop, turn_card)

            # Dangerous runout: give up
            for tag in runout_tags:
                if tag in plan.give_up_runouts:
                    return "check"

            # Continue runouts: must match one (unless "any" or blank)
            if plan.continue_runouts and "any" not in plan.continue_runouts:
                has_continue = any(tag in plan.continue_runouts for tag in runout_tags)
                if not has_continue and "blank" not in runout_tags:
                    return "check"

        return plan.turn_action if plan.turn_action else "check"

    if street == "river":
        # Global give-up override
        if plan.give_up_runouts and "any" in plan.give_up_runouts:
            return "check"

        if river_card is not None and board is not None:
            # board[:4] is flop+turn; river_card is the new card
            turn_board = board[:4]
            runout_tags = analyze_runout(turn_board, river_card)

            # Dangerous runout: give up
            for tag in runout_tags:
                if tag in plan.give_up_runouts:
                    return "check"

            action = plan.river_action if plan.river_action else "check"
            if action == "bluff":
                # Only bluff on safe runouts; back off on scary/draw-completing cards
                if "scary_card" in runout_tags or "flush_completes" in runout_tags:
                    return "check"
            return action

        action = plan.river_action if plan.river_action else "check"
        if action == "bluff" and plan.bluff_line is None:
            # river_action was set to "bluff" but no bluff_line was assigned;
            # fall back to a passive check rather than attempting an unplanned bluff
            return "check"
        return action

    return "check"
