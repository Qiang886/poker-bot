"""Post-flop decision engine."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.card import Card
from src.position import Position, is_in_position
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand
from src.board_analysis import analyze_board, analyze_range_advantage, RangeAdvantage, NutAdvantage
from src.sizing import calculate_sizing, SizingProfile
from src.barrel_plan import BarrelPlan, ValueLine, create_barrel_plan, get_current_action
from src.weighted_range import ComboWeight
from src.opponent import VillainProfile
from src.multiway import adjust_for_multiway
from src.sizing_tell import SizingTellInterpreter
from src.outs_calculator import OutsCalculator, format_outs_summary


@dataclass
class PostflopDecision:
    action: str       # "fold", "check", "call", "bet", "raise", "all-in"
    amount: float
    reasoning: str
    confidence: float  # 0-1


# Minimum SPR to consider a raise vs polarized sizing (below this just call or shove)
_MIN_SPR_FOR_POLARIZED_RAISE = 3


class PostflopEngine:
    """Rule-based postflop decision engine."""

    def __init__(self) -> None:
        self._sizing_tell = SizingTellInterpreter()
        self._outs_calc = OutsCalculator()

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
        sizing_ratio: float = 0.0,
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
            # Compute sizing ratio for facing-bet analysis
            actual_ratio = sizing_ratio if sizing_ratio > 0 else (to_call / pot if pot > 0 else 0.5)
            base_decision = self._facing_bet_decision(
                hero_hand, board, pot, to_call, hero_stack,
                hand_strength, board_texture, range_adv, position, villain_profile,
                is_multiway, multiway_adj, villain_range, street, actual_ratio,
                barrel_plan=barrel_plan,
            )
        else:
            base_decision = self._first_to_act_decision(
                hero_hand, board, pot, hero_stack,
                hand_strength, board_texture, range_adv, position, street,
                villain_profile, barrel_plan, is_multiway, multiway_adj, villain_stack,
            )

        return self._apply_villain_exploit(
            base_decision, villain_profile, hand_strength,
            board_texture, pot, to_call, hero_stack, street,
            is_facing_bet=(to_call > 0),
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
        villain_range: List[ComboWeight],
        street: str,
        sizing_ratio: float,
        barrel_plan: Optional[BarrelPlan] = None,
    ) -> PostflopDecision:
        spr = hero_stack / pot if pot > 0 else 10.0
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        equity = hand_strength.equity_bucket
        made = hand_strength.made_hand
        draw = hand_strength.draw

        # --- Execute check-raise plan if barrel_plan signals CHECK_RAISE ---
        if barrel_plan is not None and barrel_plan.value_line == ValueLine.CHECK_RAISE:
            raise_size = min(hero_stack, pot * 0.85 + to_call * 3)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"Check-raise: {made.name} executing planned check-raise",
                confidence=0.85,
            )

        # --- Sizing tell: interpret villain's bet ---
        sizing_interp = self._sizing_tell.interpret(
            action="bet",
            sizing_ratio=sizing_ratio,
            street=street,
            board_texture=board_texture,
            villain_profile=villain_profile,
        )
        sizing_note = f"[sizing: {sizing_interp.polarization}] "

        # --- Outs analysis for draw hands ---
        outs_note = ""
        outs_analysis = None
        has_draw = draw != DrawType.NONE
        if has_draw and len(board) >= 3:
            outs_analysis = self._outs_calc.calculate_outs(
                hero_hand, board, villain_range, simulations_per_out=150
            )
            if outs_analysis.total_clean > 0 or outs_analysis.total_dirty > 0:
                outs_note = f" | outs: {format_outs_summary(outs_analysis)}"
                # Override equity estimate with true equity from outs analysis
                if outs_analysis.true_equity > equity:
                    equity = outs_analysis.true_equity

        # Boost equity estimate for draws (keep as safety floor)
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
                    reasoning=f"{sizing_note}{made.name}: low SPR, shove for value",
                    confidence=0.95,
                )
            raise_size = min(hero_stack, pot * 0.75 + to_call * 2.5)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"{sizing_note}{made.name}: raise for value",
                confidence=0.90,
            )

        # Strong value: call or raise
        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            if spr < 2:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"{sizing_note}{made.name}: committing stack",
                    confidence=0.85,
                )
            if ev_call > 0:
                # If villain is polarized (large/overbet), consider raise with top hands
                if sizing_interp.polarization in ("polarized", "very_polarized") and spr > _MIN_SPR_FOR_POLARIZED_RAISE:
                    raise_size = min(hero_stack, pot * 0.75 + to_call * 2.5)
                    return PostflopDecision(
                        action="raise",
                        amount=raise_size,
                        reasoning=(
                            f"{sizing_note}{made.name}: raise vs polarized sizing "
                            f"(value_pct={sizing_interp.estimated_value_pct:.0%})"
                        ),
                        confidence=0.78,
                    )
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"{sizing_note}{made.name}: positive EV call (equity={equity:.0%}, pot_odds={pot_odds:.0%})",
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
                        reasoning=f"{sizing_note}Semi-bluff raise: {draw.name}{outs_note}",
                        confidence=0.70,
                    )
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"{sizing_note}Semi-bluff call: {draw.name} has equity {equity:.0%}{outs_note}",
                    confidence=0.65,
                )

        # Positive EV call for medium strength hands
        if ev_call > 0 and hand_strength.has_showdown_value:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"{sizing_note}Call: positive EV ({ev_call:.2f}), {made.name}",
                confidence=0.60,
            )

        # Marginal draws with odds
        if equity > pot_odds and draw != DrawType.NONE:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"{sizing_note}Drawing call: equity {equity:.0%} > pot odds {pot_odds:.0%}{outs_note}",
                confidence=0.55,
            )

        # Fold — if villain is very polarized and we have showdown value, pot-odds bluff-catch
        if (
            sizing_interp.polarization in ("polarized", "very_polarized")
            and hand_strength.has_showdown_value
            and equity >= pot_odds * 0.90
        ):
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=(
                    f"{sizing_note}Bluff-catch: villain polarized "
                    f"(bluff_pct≈{sizing_interp.estimated_bluff_pct:.0%}), "
                    f"showdown value with {made.name}"
                ),
                confidence=0.52,
            )

        return PostflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"{sizing_note}Fold: equity {equity:.0%} < pot odds {pot_odds:.0%}, {made.name}",
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
        in_position = pos_label == "ip"

        # Check-raise evaluation (OOP only, before barrel plan)
        if not in_position:
            cr_decision = self._evaluate_check_raise(
                hand_strength, board_texture, range_adv,
                spr, pot, villain_profile, street, is_multiway,
            )
            if cr_decision is not None:
                return cr_decision

        # Use barrel plan if available
        if barrel_plan is not None:
            turn_card = board[3] if len(board) >= 4 else None
            river_card = board[4] if len(board) >= 5 else None
            planned_action = get_current_action(
                barrel_plan, street,
                turn_card=turn_card,
                river_card=river_card,
                board=board,
            )
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

    def _evaluate_check_raise(
        self,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        spr: float,
        pot: float,
        villain_profile: Optional[VillainProfile],
        street: str,
        is_multiway: bool,
    ) -> Optional[PostflopDecision]:
        """Evaluate whether to plan a check-raise when OOP.

        Returns a PostflopDecision with action="check" (and check-raise intent
        embedded in the reasoning) when conditions are met, or None otherwise.
        """
        if is_multiway:
            return None

        made = hand_strength.made_hand
        draw = hand_strength.draw

        # Check-raise for value: monster hands OOP
        if made >= MadeHandType.TRIPS_SET:
            if spr > 2:
                if villain_profile is not None:
                    cbet_freq = getattr(villain_profile.stats, "cbet_flop", 0.6)
                    if cbet_freq < 0.40:
                        return None  # villain rarely cbets; no opportunity
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Check-raise plan: {made.name} OOP, trap for value",
                    confidence=0.82,
                )

        # Check-raise for value: top two pair on wet board
        if made >= MadeHandType.TOP_TWO_PAIR and board_texture.wetness >= 6:
            if spr > 2:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Check-raise plan: {made.name} on wet board, protect + build pot",
                    confidence=0.75,
                )

        # Check-raise as semi-bluff (flop only)
        if street == "flop" and draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT):
            if spr > 3:
                if villain_profile is not None:
                    # Use fold_to_flop_cbet as a proxy for villain's likelihood
                    # of folding to a check-raise semi-bluff.
                    fold_cbet = getattr(villain_profile.stats, "fold_to_flop_cbet", 0.5)
                    if fold_cbet > 0.35:
                        return PostflopDecision(
                            action="check",
                            amount=0.0,
                            reasoning=f"Check-raise bluff plan: {draw.name} OOP, semi-bluff",
                            confidence=0.60,
                        )
                else:
                    # No villain data: nut draw on flop is default check-raise spot OOP
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Check-raise bluff plan: {draw.name} OOP, semi-bluff",
                        confidence=0.58,
                    )

        return None

    def _apply_villain_exploit(
        self,
        base_decision: PostflopDecision,
        villain_profile: Optional[VillainProfile],
        hand_strength: HandStrength,
        board_texture,
        pot: float,
        to_call: float,
        hero_stack: float,
        street: str,
        is_facing_bet: bool,
    ) -> PostflopDecision:
        """Adjust a base decision based on villain player-type exploitation."""
        if villain_profile is None or villain_profile.stats.hands_played < 30:
            return base_decision

        player_type = villain_profile.classify()
        made = hand_strength.made_hand
        s = villain_profile.stats

        # ── vs FISH (high VPIP, passive) ──
        if player_type == "fish":
            # Thin value bet: fish calls down with weak hands
            if base_decision.action == "check" and hand_strength.has_showdown_value and not is_facing_bet:
                if made >= MadeHandType.MIDDLE_PAIR:
                    bet_amount = min(pot * 0.5, hero_stack)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Exploit fish: thin value bet with {made.name}",
                        confidence=0.70,
                    )
            # Fish large bet/raise = real hand (fish rarely bluff)
            if is_facing_bet and base_decision.action == "call":
                if made <= MadeHandType.MIDDLE_PAIR and to_call > pot * 0.5:
                    return PostflopDecision(
                        action="fold",
                        amount=0.0,
                        reasoning=f"Exploit fish: large bet = real hand, fold {made.name}",
                        confidence=0.65,
                    )
            # Don't bluff fish (they don't fold)
            if base_decision.action == "bet" and "Bluff" in base_decision.reasoning:
                fold_freq = getattr(s, "fold_to_flop_cbet", 0.4)
                if fold_freq < 0.35:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Exploit fish: don't bluff, fold_to_cbet={fold_freq:.0%}",
                        confidence=0.70,
                    )

        # ── vs NIT (tight passive) ──
        elif player_type == "nit":
            # Cheap bluff: nit folds too much
            if base_decision.action == "check" and not is_facing_bet:
                fold_freq = getattr(s, "fold_to_flop_cbet", 0.5)
                if fold_freq > 0.55:
                    bet_amount = min(pot * 0.33, hero_stack)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Exploit nit: high fold freq ({fold_freq:.0%}), cheap bluff",
                        confidence=0.65,
                    )
            # Nit large bet/raise = nuts
            if is_facing_bet and base_decision.action == "call":
                af = getattr(s, "aggression_factor", 1.0)
                if af < 1.5 and to_call > pot * 0.6:
                    if made < MadeHandType.TOP_PAIR_TOP_KICKER:
                        return PostflopDecision(
                            action="fold",
                            amount=0.0,
                            reasoning=f"Exploit nit: passive player large bet = nuts, fold",
                            confidence=0.75,
                        )

        # ── vs LAG (loose aggressive) ──
        elif player_type == "LAG":
            # Wider call-down: LAG bluffs frequently
            if is_facing_bet and base_decision.action == "fold":
                if hand_strength.has_showdown_value and made >= MadeHandType.MIDDLE_PAIR:
                    af = getattr(s, "aggression_factor", 2.0)
                    if af > 2.5:
                        return PostflopDecision(
                            action="call",
                            amount=to_call,
                            reasoning=f"Exploit LAG: wide call-down, AF={af:.1f}, {made.name}",
                            confidence=0.60,
                        )
            # Don't bluff a re-bluffer
            if base_decision.action == "bet" and "Bluff" in base_decision.reasoning:
                af = getattr(s, "aggression_factor", 2.0)
                if af > 3.0:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Exploit LAG: don't bluff aggressive player (AF={af:.1f})",
                        confidence=0.65,
                    )
            # Trap LAG with strong hands
            if not is_facing_bet and made >= MadeHandType.TRIPS_SET:
                af = getattr(s, "aggression_factor", 2.0)
                if af > 3.0:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Exploit LAG: trap with {made.name}, let LAG bluff (AF={af:.1f})",
                        confidence=0.70,
                    )

        # ── vs TAG (balanced) ──
        elif player_type == "TAG":
            # River bluff when TAG over-folds river (no blocker data available)
            if street == "river" and not is_facing_bet and base_decision.action == "check":
                fold_to_river = getattr(s, "fold_to_river_cbet", 0.45)
                if fold_to_river > 0.60:
                    bet_amount = min(pot * 0.75, hero_stack)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Exploit TAG: river bluff, fold_to_river={fold_to_river:.0%}",
                        confidence=0.55,
                    )

        return base_decision
