"""Pre-flop decision engine."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.card import Card, Rank, RANK_CHAR
from src.position import Position
from src.ranges import (
    get_rfi_range, get_3bet_range, get_4bet_range,
    get_call_rfi_range, get_call_3bet_range, get_squeeze_range,
)
from src.opponent import VillainProfile


@dataclass
class PreflopDecision:
    action: str   # "fold", "call", "raise", "3bet", "4bet", "all-in", "limp"
    amount: float
    reasoning: str


# ---------------------------------------------------------------------------
# Open-raise sizes (in big blinds) by position
# ---------------------------------------------------------------------------
_OPEN_SIZES: Dict[Position, float] = {
    Position.UTG: 2.5,
    Position.HJ:  2.5,
    Position.CO:  2.5,
    Position.BTN: 2.5,
    Position.SB:  3.0,
    Position.BB:  0.0,   # BB never opens, acts last preflop
}

# 3bet sizes (relative to the open raise, in total pot terms factor)
_3BET_IP_FACTOR = 3.0
_3BET_OOP_FACTOR = 4.0

# 4bet sizing
_4BET_FACTOR = 2.5   # of 3-bet size


class PreflopEngine:
    """Rule-based preflop decision engine."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(
        self,
        hero_hand: Tuple[Card, Card],
        position: Position,
        action_history: List[Dict],
        pot: float,
        to_call: float,
        hero_stack: float,
        num_players: int,
        villain_profiles: Dict[int, VillainProfile],
    ) -> PreflopDecision:
        """Return the best preflop action."""
        notation = self._get_hand_notation(hero_hand)
        bb = 1.0   # 1 big blind unit

        # Determine what has happened so far
        raises = [a for a in action_history if a.get("action") in ("raise", "open", "3bet", "4bet")]
        calls = [a for a in action_history if a.get("action") == "call"]
        limps = [a for a in action_history if a.get("action") == "limp"]

        # ------------------------------------------------------------------
        # Squeeze opportunity
        # ------------------------------------------------------------------
        if len(raises) == 1 and len(calls) >= 1:
            sq_range = get_squeeze_range(position)
            if notation in sq_range:
                sq_size = min(hero_stack, (raises[0].get("amount", 2.5) + sum(c.get("amount", 2.5) for c in calls)) * 2.5)
                return PreflopDecision(
                    action="3bet",
                    amount=sq_size,
                    reasoning=f"Squeeze with {notation} from {position.name}",
                )

        # ------------------------------------------------------------------
        # Facing a 3-bet (after we opened)
        # ------------------------------------------------------------------
        if len(raises) >= 2:
            villain_rfi_action = raises[0]
            villain_3bet_action = raises[-1]
            villain_pos = villain_rfi_action.get("position", Position.BTN)
            amount_3bet = villain_3bet_action.get("amount", 9.0)

            if self._should_4bet(notation, position, villain_pos):
                bet_4bet = min(hero_stack, amount_3bet * _4BET_FACTOR)
                return PreflopDecision(
                    action="4bet",
                    amount=bet_4bet,
                    reasoning=f"4bet with {notation} vs 3bet",
                )

            call_3b_range = get_call_3bet_range(position, villain_pos)
            if notation in call_3b_range:
                return PreflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Call 3bet with {notation} (in call range)",
                )

            return PreflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"Fold vs 3bet: {notation} not in 4bet/call range",
            )

        # ------------------------------------------------------------------
        # Facing an open raise (first raise only)
        # ------------------------------------------------------------------
        if len(raises) == 1 and not calls:
            villain_action = raises[0]
            villain_pos = villain_action.get("position", Position.UTG)
            amount_rfi = villain_action.get("amount", 2.5)

            if self._should_3bet(notation, position, villain_pos):
                # Use exploit adjustment if villain folds a lot to 3bets
                vp = villain_profiles.get(0)
                fold_rate = 0.65
                if vp is not None:
                    fold_rate = vp.stats.fold_to_3bet_by_position.get(villain_pos, 0.65)

                in_pos = self._is_in_position(position, villain_pos)
                factor = _3BET_IP_FACTOR if in_pos else _3BET_OOP_FACTOR
                bet_3bet = min(hero_stack, amount_rfi * factor)
                return PreflopDecision(
                    action="3bet",
                    amount=bet_3bet,
                    reasoning=f"3bet with {notation} from {position.name} vs {villain_pos.name} open (villain fold rate {fold_rate:.0%})",
                )

            if self._should_call_rfi(notation, position, villain_pos):
                return PreflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Call open with {notation} from {position.name}",
                )

            return PreflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"Fold to open: {notation} not in 3bet/call range from {position.name}",
            )

        # ------------------------------------------------------------------
        # Facing limps only (no raise yet)
        # ------------------------------------------------------------------
        if limps and not raises:
            # Iso-raise strong hands
            rfi = get_rfi_range(position)
            if notation in rfi:
                iso_size = self._calculate_open_size(position, hero_stack) + len(limps) * bb
                return PreflopDecision(
                    action="raise",
                    amount=min(hero_stack, iso_size),
                    reasoning=f"Iso-raise limpers with {notation}",
                )
            return PreflopDecision(
                action="call",
                amount=bb,
                reasoning=f"Over-limp with {notation}",
            )

        # ------------------------------------------------------------------
        # No action yet (RFI opportunity)
        # ------------------------------------------------------------------
        if self._is_in_rfi_range(notation, position):
            open_size = self._calculate_open_size(position, hero_stack)
            return PreflopDecision(
                action="raise",
                amount=open_size,
                reasoning=f"Open raise {notation} from {position.name}",
            )

        # BB special case: check if no raise
        if position == Position.BB and to_call == 0:
            return PreflopDecision(
                action="check",
                amount=0.0,
                reasoning=f"BB checks option with {notation}",
            )

        return PreflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"Fold: {notation} not in RFI range for {position.name}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hand_notation(self, hand: Tuple[Card, Card]) -> str:
        """Convert two cards to hand notation: 'AA', 'AKs', 'AKo'."""
        c1, c2 = hand
        if c1.rank < c2.rank:
            c1, c2 = c2, c1
        r1_char = RANK_CHAR[c1.rank]
        r2_char = RANK_CHAR[c2.rank]
        if c1.rank == c2.rank:
            return f"{r1_char}{r2_char}"
        if c1.suit == c2.suit:
            return f"{r1_char}{r2_char}s"
        return f"{r1_char}{r2_char}o"

    def _is_in_rfi_range(self, notation: str, position: Position) -> bool:
        return notation in get_rfi_range(position)

    def _should_3bet(self, notation: str, hero_pos: Position, villain_pos: Position) -> bool:
        return notation in get_3bet_range(hero_pos, villain_pos)

    def _should_4bet(self, notation: str, hero_pos: Position, villain_pos: Position) -> bool:
        return notation in get_4bet_range(hero_pos)

    def _should_call_rfi(self, notation: str, hero_pos: Position, villain_pos: Position) -> bool:
        return notation in get_call_rfi_range(hero_pos, villain_pos)

    def _is_in_position(self, hero_pos: Position, villain_pos: Position) -> bool:
        """Preflop position: higher seat number = later action = in position."""
        order = [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]
        # Postflop: BTN is in position on everyone except BB/SB
        postflop = [Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO, Position.BTN]
        return postflop.index(hero_pos) > postflop.index(villain_pos)

    def _calculate_open_size(self, position: Position, stack: float) -> float:
        """Return open-raise size in chips (assuming BB=1)."""
        base = _OPEN_SIZES.get(position, 2.5)
        return min(stack, base)
