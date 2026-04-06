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
                base = PreflopDecision(
                    action="3bet",
                    amount=sq_size,
                    reasoning=f"Squeeze with {notation} from {position.name}",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
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
                base = PreflopDecision(
                    action="4bet",
                    amount=bet_4bet,
                    reasoning=f"4bet with {notation} vs 3bet",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
                )

            call_3b_range = get_call_3bet_range(position, villain_pos)
            if notation in call_3b_range:
                base = PreflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Call 3bet with {notation} (in call range)",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
                )

            base = PreflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"Fold vs 3bet: {notation} not in 4bet/call range",
            )
            return self._apply_preflop_exploit(
                base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
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
                base = PreflopDecision(
                    action="3bet",
                    amount=bet_3bet,
                    reasoning=f"3bet with {notation} from {position.name} vs {villain_pos.name} open (villain fold rate {fold_rate:.0%})",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
                )

            if self._should_call_rfi(notation, position, villain_pos):
                base = PreflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Call open with {notation} from {position.name}",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
                )

            base = PreflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"Fold to open: {notation} not in 3bet/call range from {position.name}",
            )
            return self._apply_preflop_exploit(
                base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
            )

        # ------------------------------------------------------------------
        # Facing limps only (no raise yet)
        # ------------------------------------------------------------------
        if limps and not raises:
            # Iso-raise strong hands
            rfi = get_rfi_range(position)
            if notation in rfi:
                iso_size = self._calculate_open_size(position, hero_stack) + len(limps) * bb
                base = PreflopDecision(
                    action="raise",
                    amount=min(hero_stack, iso_size),
                    reasoning=f"Iso-raise limpers with {notation}",
                )
                return self._apply_preflop_exploit(
                    base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
                )
            base = PreflopDecision(
                action="call",
                amount=bb,
                reasoning=f"Over-limp with {notation}",
            )
            return self._apply_preflop_exploit(
                base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
            )

        # ------------------------------------------------------------------
        # No action yet (RFI opportunity)
        # ------------------------------------------------------------------
        if self._is_in_rfi_range(notation, position):
            open_size = self._calculate_open_size(position, hero_stack)
            base = PreflopDecision(
                action="raise",
                amount=open_size,
                reasoning=f"Open raise {notation} from {position.name}",
            )
            return self._apply_preflop_exploit(
                base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
            )

        # BB special case: check if no raise
        if position == Position.BB and to_call == 0:
            return PreflopDecision(
                action="check",
                amount=0.0,
                reasoning=f"BB checks option with {notation}",
            )

        base = PreflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"Fold: {notation} not in RFI range for {position.name}",
        )
        return self._apply_preflop_exploit(
            base, notation, position, action_history, pot, to_call, hero_stack, villain_profiles,
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

    # ------------------------------------------------------------------
    # Preflop exploit
    # ------------------------------------------------------------------

    def _apply_preflop_exploit(
        self,
        base_decision: PreflopDecision,
        notation: str,
        position: Position,
        action_history: List[Dict],
        pot: float,
        to_call: float,
        hero_stack: float,
        villain_profiles: Dict[int, "VillainProfile"],
    ) -> PreflopDecision:
        """Adjust preflop decision based on villain player type.

        Requires at least 30 hand sample before exploiting.
        """
        main_villain = self._get_main_villain(villain_profiles, action_history)
        if main_villain is None or main_villain.stats.hands_played < 30:
            return base_decision

        player_type = main_villain.classify()
        raises = [a for a in action_history if a.get("action") in ("raise", "open", "3bet", "4bet")]
        limps = [a for a in action_history if a.get("action") == "limp"]

        # ─── vs FISH (VPIP > 40%) ───
        if player_type == "fish":
            # Wider open range: fish in blinds → expand to marginal hands
            if base_decision.action == "fold" and not action_history:
                if position in (Position.CO, Position.BTN, Position.SB):
                    if self._is_marginal_open(notation):
                        open_size = self._calculate_open_size(position, hero_stack)
                        return PreflopDecision(
                            action="raise",
                            amount=open_size,
                            reasoning=f"Exploit fish: wider open with {notation} from {position.name}",
                        )

            # Larger open sizing vs fish (build bigger pot)
            if base_decision.action == "raise" and not action_history:
                fish_size = min(hero_stack, 3.5)
                return PreflopDecision(
                    action="raise",
                    amount=fish_size,
                    reasoning=f"Exploit fish: larger open sizing {fish_size}BB with {notation}",
                )

            # Larger iso-raise over limpers
            if limps and base_decision.action == "raise":
                iso_size = min(hero_stack, 4.0 + len(limps) * 1.5)
                return PreflopDecision(
                    action="raise",
                    amount=iso_size,
                    reasoning=f"Exploit fish: large iso-raise {iso_size:.1f}BB over {len(limps)} limps",
                )

            # Value 3bet wider: JJ/TT/AQo vs fish (fish calls too wide)
            if raises and base_decision.action == "call":
                value_3bet_vs_fish = {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}
                if notation in value_3bet_vs_fish:
                    amount_rfi = raises[0].get("amount", 2.5)
                    bet_3bet = min(hero_stack, amount_rfi * 3.5)
                    return PreflopDecision(
                        action="3bet",
                        amount=bet_3bet,
                        reasoning=f"Exploit fish: value 3bet {notation} (fish calls too wide)",
                    )

            # Don't bluff 3bet fish – fish doesn't fold
            if base_decision.action == "3bet":
                bluff_3bets = {"A5s", "A4s", "A3s", "A2s", "76s", "65s", "54s"}
                if notation in bluff_3bets:
                    return PreflopDecision(
                        action="call",
                        amount=to_call,
                        reasoning=f"Exploit fish: don't bluff 3bet fish, call instead with {notation}",
                    )

        # ─── vs NIT (VPIP < 15%) ───
        elif player_type == "nit":
            # Steal blinds aggressively vs nit
            if base_decision.action == "fold" and not action_history:
                if position in (Position.CO, Position.BTN, Position.SB):
                    if self._is_stealable(notation):
                        steal_size = min(hero_stack, 2.0)
                        return PreflopDecision(
                            action="raise",
                            amount=steal_size,
                            reasoning=f"Exploit nit: steal with {notation}, nit folds too much",
                        )

            # Tighten call/3bet range vs nit open
            if raises and base_decision.action in ("call", "3bet"):
                nit_3bet_range = {"AA", "KK", "QQ", "AKs", "AKo"}
                nit_call_range = {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}

                if base_decision.action == "3bet" and notation not in nit_3bet_range:
                    if notation in nit_call_range:
                        return PreflopDecision(
                            action="call",
                            amount=to_call,
                            reasoning=f"Exploit nit: downgrade {notation} from 3bet to call vs nit open",
                        )
                    return PreflopDecision(
                        action="fold",
                        amount=0.0,
                        reasoning=f"Exploit nit: fold {notation} vs nit open (nit range too strong)",
                    )

                if (
                    base_decision.action == "call"
                    and notation not in nit_call_range
                    and notation not in nit_3bet_range
                ):
                    return PreflopDecision(
                        action="fold",
                        amount=0.0,
                        reasoning=f"Exploit nit: fold {notation} vs nit open",
                    )

            # Smaller open sizing vs nit (nit won't 3bet light)
            if base_decision.action == "raise" and not action_history:
                return PreflopDecision(
                    action="raise",
                    amount=min(hero_stack, 2.0),
                    reasoning=f"Exploit nit: min-open {notation} from {position.name}, nit won't 3bet light",
                )

        # ─── vs LAG (VPIP > 30%, AF > 3) ───
        elif player_type == "LAG":
            # 4bet bluff vs LAG 3bet (LAG 3bet range is wide)
            if len(raises) >= 2 and base_decision.action == "fold":
                lag_4bet_bluffs = {"A5s", "A4s", "A3s", "A2s", "KJs", "QJs"}
                if notation in lag_4bet_bluffs:
                    amount_3bet = raises[-1].get("amount", 9.0)
                    bet_4bet = min(hero_stack, amount_3bet * 2.5)
                    return PreflopDecision(
                        action="4bet",
                        amount=bet_4bet,
                        reasoning=f"Exploit LAG: 4bet bluff {notation} vs LAG 3bet (their 3bet range is wide)",
                    )

            # Trap AA/KK vs LAG 3bet: flat call instead of 4bet
            if len(raises) >= 2 and base_decision.action == "4bet":
                trap_hands = {"AA", "KK"}
                if notation in trap_hands:
                    return PreflopDecision(
                        action="call",
                        amount=to_call,
                        reasoning=f"Exploit LAG: trap with {notation}, flat call LAG's 3bet",
                    )

        # ─── vs TAG (balanced, minimal exploit) ───
        elif player_type == "TAG":
            avg_fold = sum(main_villain.stats.fold_to_3bet_by_position.values()) / max(
                len(main_villain.stats.fold_to_3bet_by_position), 1
            )
            if avg_fold > 0.70:
                if raises and base_decision.action == "fold":
                    tag_bluff_3bets = {"A5s", "A4s", "76s", "87s", "65s", "98s"}
                    if notation in tag_bluff_3bets and position in (Position.CO, Position.BTN, Position.SB):
                        amount_rfi = raises[0].get("amount", 2.5)
                        bet_3bet = min(hero_stack, amount_rfi * 3.0)
                        return PreflopDecision(
                            action="3bet",
                            amount=bet_3bet,
                            reasoning=f"Exploit TAG: 3bet bluff {notation}, TAG folds to 3bet {avg_fold:.0%}",
                        )

        return base_decision

    def _is_marginal_open(self, notation: str) -> bool:
        """Check if notation is a marginal open hand (not in standard chart but playable vs fish)."""
        marginal = {
            "55", "44", "33", "22",
            "K9o", "K8o", "Q9o", "Q8o", "J9o", "J8o", "T9o", "T8o",
            "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
            "Q6s", "Q5s", "Q4s",
            "J7s", "J6s",
            "T7s", "T6s", "96s", "85s", "74s", "63s", "52s",
        }
        return notation in marginal

    def _is_stealable(self, notation: str) -> bool:
        """Check if hand is suitable for stealing vs nit.

        Uses a blacklist of the absolute worst hands (trash hands with no
        playability). Everything else is considered stealable given that
        vs a nit who folds too much, even weak hands have +EV open-steals.
        """
        not_stealable = {"32o", "42o", "52o", "62o", "72o", "43o", "53o", "63o", "73o"}
        return notation not in not_stealable

    def _get_main_villain(
        self,
        villain_profiles: Dict[int, "VillainProfile"],
        action_history: List[Dict],
    ) -> "Optional[VillainProfile]":
        """Find the main villain's profile from action history."""
        for event in action_history:
            actor = event.get("actor", event.get("villain_id"))
            if actor is not None and actor in villain_profiles:
                return villain_profiles[actor]
        return villain_profiles.get(0)
