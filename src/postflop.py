"""Post-flop decision engine."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.card import Card
from src.position import Position, is_in_position
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand
from src.board_analysis import analyze_board, analyze_range_advantage, RangeAdvantage, NutAdvantage
from src.sizing import calculate_sizing, SizingProfile
from src.barrel_plan import BarrelPlan, create_barrel_plan, get_current_action
from src.weighted_range import ComboWeight
from src.opponent import VillainProfile
from src.multiway import adjust_for_multiway


@dataclass
class PostflopDecision:
    action: str       # "fold", "check", "call", "bet", "raise", "all-in"
    amount: float
    reasoning: str
    confidence: float  # 0-1


class PostflopEngine:
    """Rule-based postflop decision engine."""

    def decide(
        self,
        hero_hand: Tuple[Card, Card],
        board: List[Card],
        pot: float,
        to_call: float,
        hero_stack: float,
        villain_stack: float,
        position: Position,
        street: str,
        villain_profile: Optional[VillainProfile],
        hero_range: List[ComboWeight],
        villain_range: List[ComboWeight],
        barrel_plan: Optional[BarrelPlan],
        is_multiway: bool = False,
    ) -> PostflopDecision:
        """Top-level postflop decision entry-point."""
        hand_strength = classify_hand(hero_hand, board)
        board_texture = analyze_board(board)
        range_adv = analyze_range_advantage(hero_range, villain_range, board)

        # Determine position label
        in_pos = True  # default: assume in position
        if villain_range:
            # Infer from a representative position comparison
            pass
        pos_label = "ip" if position in (Position.BTN, Position.CO) else "oop"

        spr = hero_stack / pot if pot > 0 else 10.0
        multiway_adj = adjust_for_multiway(hand_strength, 2 if not is_multiway else 3, board_texture)

        if to_call > 0:
            return self._facing_bet_decision(
                hero_hand, board, pot, to_call, hero_stack,
                hand_strength, board_texture, range_adv, position, villain_profile,
                is_multiway, multiway_adj,
            )
        return self._first_to_act_decision(
            hero_hand, board, pot, hero_stack,
            hand_strength, board_texture, range_adv, position, street,
            villain_profile, barrel_plan, is_multiway, multiway_adj, villain_stack,
        )

    # ------------------------------------------------------------------
    # Facing a bet
    # ------------------------------------------------------------------

    def _facing_bet_decision(
        self,
        hero_hand: Tuple[Card, Card],
        board: List[Card],
        pot: float,
        to_call: float,
        hero_stack: float,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        villain_profile: Optional[VillainProfile],
        is_multiway: bool,
        multiway_adj,
    ) -> PostflopDecision:
        spr = hero_stack / pot if pot > 0 else 10.0
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        equity = hand_strength.equity_bucket
        made = hand_strength.made_hand
        draw = hand_strength.draw

        # Boost equity estimate for draws
        if draw == DrawType.FLUSH_DRAW_NUT:
            equity = max(equity, 0.38)
        elif draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
            equity = max(equity, 0.50)
        elif draw == DrawType.OESD:
            equity = max(equity, 0.32)
        elif draw == DrawType.FLUSH_DRAW_LOW:
            equity = max(equity, 0.34)
        elif draw == DrawType.GUTSHOT:
            equity = max(equity, 0.20)

        ev_call = self._calculate_call_ev(to_call, pot, equity)

        # ------------------------------------------------------------------
        # Monster hands: raise or call depending on stack depth
        # ------------------------------------------------------------------
        if made >= MadeHandType.TRIPS_SET:
            if spr < 3:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"{made.name}: low SPR, shove for value",
                    confidence=0.95,
                )
            raise_size = min(hero_stack, pot * 0.75 + to_call * 2.5)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"{made.name}: raise for value",
                confidence=0.90,
            )

        # Strong value: call or raise
        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            if spr < 2:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"{made.name}: committing stack",
                    confidence=0.85,
                )
            if ev_call > 0:
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"{made.name}: positive EV call (equity={equity:.0%}, pot_odds={pot_odds:.0%})",
                    confidence=0.80,
                )

        # Nut draws: call or raise as semi-bluff
        if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT):
            if equity > pot_odds + 0.05:
                if not is_multiway and spr > 2:
                    raise_size = min(hero_stack, pot * 0.85 + to_call * 2)
                    return PostflopDecision(
                        action="raise",
                        amount=raise_size,
                        reasoning=f"Semi-bluff raise: {draw.name}",
                        confidence=0.70,
                    )
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Semi-bluff call: {draw.name} has equity {equity:.0%}",
                    confidence=0.65,
                )

        # Positive EV call for medium strength hands
        if ev_call > 0 and hand_strength.has_showdown_value:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"Call: positive EV ({ev_call:.2f}), {made.name}",
                confidence=0.60,
            )

        # Marginal draws with odds
        if equity > pot_odds and draw != DrawType.NONE:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"Drawing call: equity {equity:.0%} > pot odds {pot_odds:.0%}",
                confidence=0.55,
            )

        # Fold
        return PostflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"Fold: equity {equity:.0%} < pot odds {pot_odds:.0%}, {made.name}",
            confidence=0.70,
        )

    # ------------------------------------------------------------------
    # First to act (no bet facing)
    # ------------------------------------------------------------------

    def _first_to_act_decision(
        self,
        hero_hand: Tuple[Card, Card],
        board: List[Card],
        pot: float,
        hero_stack: float,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        street: str,
        villain_profile: Optional[VillainProfile],
        barrel_plan: Optional[BarrelPlan],
        is_multiway: bool,
        multiway_adj,
        villain_stack: float,
    ) -> PostflopDecision:
        spr = hero_stack / pot if pot > 0 else 10.0
        pos_label = "ip" if position in (Position.BTN, Position.CO, Position.HJ) else "oop"

        # Use barrel plan if available
        if barrel_plan is not None:
            planned_action = get_current_action(barrel_plan, street)
            if planned_action == "bet" or planned_action == "bluff" or planned_action == "barrel":
                if self._should_value_bet(hand_strength, board_texture, range_adv, position, spr):
                    sizing = calculate_sizing(
                        hand_strength, board_texture, range_adv, street,
                        spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
                    )
                    bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                    return PostflopDecision(
                        action="bet" if bet_amount < hero_stack else "all-in",
                        amount=bet_amount,
                        reasoning=f"Value bet ({sizing.reasoning}): {hand_strength.made_hand.name}",
                        confidence=0.80,
                    )
                if self._should_bluff(hand_strength, board_texture, range_adv, position, spr, pot):
                    sizing = calculate_sizing(
                        hand_strength, board_texture, range_adv, street,
                        spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
                    )
                    bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Bluff ({sizing.reasoning}): {hand_strength.draw.name}",
                        confidence=0.55,
                    )
            # Planned check
            if planned_action in ("check", "give_up"):
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Following barrel plan: {planned_action}",
                    confidence=0.65,
                )

        # No plan: decide from scratch
        if self._should_value_bet(hand_strength, board_texture, range_adv, position, spr):
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, street,
                spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
            )
            if is_multiway and not multiway_adj.bluff_allowed:
                if hand_strength.made_hand < MadeHandType.TOP_PAIR_TOP_KICKER + multiway_adj.value_threshold_adjustment:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning="Multiway: tightening value range, checking instead",
                        confidence=0.65,
                    )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot * multiway_adj.bet_frequency_multiplier)
            if bet_amount < pot * 0.20:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning="Bet too small after multiway adjustment, check instead",
                    confidence=0.60,
                )
            return PostflopDecision(
                action="bet" if bet_amount < hero_stack else "all-in",
                amount=bet_amount,
                reasoning=f"Value bet ({sizing.reasoning}): {hand_strength.made_hand.name}",
                confidence=0.80,
            )

        if self._should_bluff(hand_strength, board_texture, range_adv, position, spr, pot):
            if is_multiway and not multiway_adj.bluff_allowed:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning="Multiway: bluffing not profitable, check",
                    confidence=0.70,
                )
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, street,
                spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
            )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Bluff ({sizing.reasoning}): {hand_strength.draw.name}",
                confidence=0.50,
            )

        return PostflopDecision(
            action="check",
            amount=0.0,
            reasoning=f"Check: {hand_strength.made_hand.name} – no clear bet",
            confidence=0.65,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_value_bet(
        self,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        spr: float,
    ) -> bool:
        made = hand_strength.made_hand
        in_pos = position in (Position.BTN, Position.CO)

        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            return True
        if made >= MadeHandType.TOP_PAIR_GOOD_KICKER and not board_texture.wet_board_risky()\
                if hasattr(board_texture, 'wet_board_risky') else True:
            return True
        if made >= MadeHandType.TOP_PAIR_GOOD_KICKER:
            return True
        if made >= MadeHandType.OVERPAIR_SMALL and spr < 6:
            return True
        if made >= MadeHandType.MIDDLE_PAIR and in_pos and board_texture.wetness < 5:
            return True
        return False

    def _should_bluff(
        self,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        spr: float,
        pot: float,
    ) -> bool:
        draw = hand_strength.draw
        in_pos = position in (Position.BTN, Position.CO)
        has_adv = range_adv.nut_advantage == NutAdvantage.HERO

        if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
            return True
        if draw == DrawType.OESD and has_adv:
            return True
        if draw == DrawType.FLUSH_DRAW_LOW and in_pos and has_adv:
            return True
        if hand_strength.made_hand == MadeHandType.ACE_HIGH and has_adv and in_pos:
            return True
        return False

    def _calculate_call_ev(self, to_call: float, pot: float, equity: float) -> float:
        """Simple EV: equity * (pot + to_call) - to_call."""
        return equity * (pot + to_call) - to_call
