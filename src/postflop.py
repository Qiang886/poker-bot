"""Post-flop decision engine."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.card import Card, Rank
from src.position import Position, is_in_position
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand, calculate_blocker_value
from src.board_analysis import analyze_board, analyze_range_advantage, RangeAdvantage, NutAdvantage
from src.sizing import calculate_sizing, calculate_river_sizing, SizingProfile
from src.barrel_plan import BarrelPlan, ValueLine, create_barrel_plan, get_current_action, analyze_runout
from src.weighted_range import ComboWeight
from src.opponent import VillainProfile
from src.multiway import adjust_for_multiway
from src.sizing_tell import SizingTellInterpreter
from src.outs_calculator import OutsCalculator, format_outs_summary
from src.dynamic_equity import adjust_equity_bucket


@dataclass
class PostflopDecision:
    action: str       # "fold", "check", "call", "bet", "raise", "all-in"
    amount: float
    reasoning: str
    confidence: float  # 0-1


@dataclass
class TurnChange:
    """Analysis of how the turn card changes range dynamics."""
    is_blank: bool           # low card that doesn't complete any draw
    is_overcard: bool        # higher than all flop cards
    completes_flush: bool    # third card of same suit
    completes_straight: bool # creates 3+ consecutive cards (new milestone)
    pairs_board: bool        # pairs the board for the first time
    is_scare_card: bool      # A or K on a medium/low board
    range_shift: str         # "hero_better", "villain_better", "neutral"


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

        # ------------------------------------------------------------------
        # Dynamic equity adjustment (Improvement 3)
        # ------------------------------------------------------------------
        villain_type = "unknown"
        if villain_profile is not None and villain_profile.stats.hands_played >= 30:
            villain_type = villain_profile.classify()

        turn_blank = True
        river_blank = True
        if len(board) >= 4:
            turn_tags = analyze_runout(board[:3], board[3])
            turn_blank = "blank" in turn_tags
        if len(board) >= 5:
            river_tags = analyze_runout(board[:4], board[4])
            river_blank = "blank" in river_tags

        adjusted_equity = adjust_equity_bucket(
            hand_strength, board_texture, villain_type, street,
            turn_is_blank=turn_blank, river_is_blank=river_blank,
        )
        hand_strength = HandStrength(
            made_hand=hand_strength.made_hand,
            draw=hand_strength.draw,
            equity_bucket=adjusted_equity,
            is_vulnerable=hand_strength.is_vulnerable,
            has_showdown_value=hand_strength.has_showdown_value,
        )

        if to_call > 0:
            # Compute sizing ratio for facing-bet analysis
            actual_ratio = sizing_ratio if sizing_ratio > 0 else (to_call / pot if pot > 0 else 0.5)

            # River gets independent decision framework
            if street == "river":
                return self._river_facing_bet(
                    hero_hand, board, pot, to_call, hero_stack,
                    hand_strength, board_texture, range_adv, position,
                    villain_profile, is_multiway, villain_range, actual_ratio,
                )

            # Turn gets independent decision framework (Improvement 1)
            if street == "turn":
                turn_card = board[3] if len(board) >= 4 else None
                turn_change = self._analyze_turn_change(board, turn_card, hero_hand)
                return self._turn_facing_bet(
                    hero_hand, board, pot, to_call, hero_stack,
                    hand_strength, board_texture, range_adv, position,
                    villain_profile, is_multiway, villain_range, actual_ratio,
                    turn_change, barrel_plan, villain_stack,
                )

            base_decision = self._facing_bet_decision(
                hero_hand, board, pot, to_call, hero_stack,
                hand_strength, board_texture, range_adv, position, villain_profile,
                is_multiway, multiway_adj, villain_range, street, actual_ratio,
                barrel_plan=barrel_plan,
            )
        else:
            # River gets independent decision framework
            if street == "river":
                return self._river_first_to_act(
                    hero_hand, board, pot, hero_stack,
                    hand_strength, board_texture, range_adv, position,
                    villain_profile, is_multiway, villain_range, villain_stack,
                )

            # Turn gets independent decision framework (Improvement 1)
            if street == "turn":
                turn_card = board[3] if len(board) >= 4 else None
                turn_change = self._analyze_turn_change(board, turn_card, hero_hand)
                return self._turn_first_to_act(
                    hero_hand, board, pot, hero_stack,
                    hand_strength, board_texture, range_adv, position,
                    villain_profile, barrel_plan, is_multiway, multiway_adj, villain_stack,
                    turn_change,
                )

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

        # Donk bet / probe bet evaluation (OOP only, Improvement 2)
        if not in_position:
            donk_decision = self._evaluate_donk_or_probe(
                hand_strength, board_texture, range_adv, street,
                pot, hero_stack, villain_profile, [], is_multiway, board,
            )
            if donk_decision is not None:
                return donk_decision

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

    # ------------------------------------------------------------------
    # River independent decision framework
    # ------------------------------------------------------------------

    def _river_first_to_act(
        self,
        hero_hand: Tuple[Card, Card],
        board: List[Card],
        pot: float,
        hero_stack: float,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        villain_profile: Optional[VillainProfile],
        is_multiway: bool,
        villain_range: List[ComboWeight],
        villain_stack: float,
    ) -> PostflopDecision:
        """River first-to-act: thin value / bluff selection / give-up."""
        made = hand_strength.made_hand
        spr = hero_stack / pot if pot > 0 else 10.0

        player_type = "TAG"
        if villain_profile is not None and villain_profile.stats.hands_played >= 30:
            player_type = villain_profile.classify()

        # ── Thin value bet decision ──
        can_value, value_sizing = self._can_thin_value_bet(
            hand_strength, villain_profile, player_type, board, pot, hero_stack, spr,
        )
        if can_value:
            bet_amount = min(hero_stack, pot * value_sizing)
            return PostflopDecision(
                action="bet" if bet_amount < hero_stack else "all-in",
                amount=bet_amount,
                reasoning=f"River value bet ({value_sizing:.0%} pot) vs {player_type}: {made.name}",
                confidence=0.80,
            )

        # ── Bluff selection ──
        if not is_multiway:
            should_bluff, bluff_sizing = self._should_river_bluff(
                hero_hand, hand_strength, villain_profile, player_type, villain_range,
                board, pot, hero_stack,
            )
            if should_bluff:
                bet_amount = min(hero_stack, pot * bluff_sizing)
                return PostflopDecision(
                    action="bet" if bet_amount < hero_stack else "all-in",
                    amount=bet_amount,
                    reasoning=f"River bluff ({bluff_sizing:.0%} pot) vs {player_type}: blocker advantage",
                    confidence=0.55,
                )

        # ── Default: check / give up ──
        return PostflopDecision(
            action="check",
            amount=0.0,
            reasoning=f"River check: {made.name} – no value bet or profitable bluff",
            confidence=0.65,
        )

    def _can_thin_value_bet(
        self,
        hand_strength: HandStrength,
        villain_profile: Optional[VillainProfile],
        player_type: str,
        board: List[Card],
        pot: float,
        hero_stack: float,
        spr: float,
    ) -> Tuple[bool, float]:
        """Return (can_value_bet, sizing_fraction).

        Determines if hero can extract value on the river and at what size.
        """
        made = hand_strength.made_hand

        # All-in territory: very low SPR
        if spr < 0.5:
            if made >= MadeHandType.MIDDLE_PAIR:
                return True, min(1.0, hero_stack / pot)

        # Monster → always value bet big
        if made >= MadeHandType.TRIPS_SET:
            return True, 1.25 if spr > 2 else 0.75

        # Strong value
        if made >= MadeHandType.TOP_TWO_PAIR:
            return True, 0.75

        # TPTK
        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            return True, 0.60

        # TPGK: villain-dependent
        if made >= MadeHandType.TOP_PAIR_GOOD_KICKER:
            if player_type == "nit":
                return False, 0.0  # nit only calls with better
            if player_type == "fish":
                return True, 0.50  # fish calls with weaker
            return True, 0.33  # thin value vs TAG/LAG

        # Middle pair: only vs fish
        if made >= MadeHandType.MIDDLE_PAIR:
            if player_type == "fish":
                return True, 0.33  # fish will call with ace-high
            return False, 0.0

        # Big overpair
        if made == MadeHandType.OVERPAIR_BIG:
            return True, 0.55

        # Small overpair: check if board has overcards
        if made == MadeHandType.OVERPAIR_SMALL:
            if board and hand_strength.is_vulnerable:
                return False, 0.0  # board has overcard; check-call or check-fold
            return True, 0.40

        return False, 0.0

    def _should_river_bluff(
        self,
        hero_hand: Tuple[Card, Card],
        hand_strength: HandStrength,
        villain_profile: Optional[VillainProfile],
        player_type: str,
        villain_range: List[ComboWeight],
        board: List[Card],
        pot: float,
        hero_stack: float,
    ) -> Tuple[bool, float]:
        """Return (should_bluff, sizing_fraction).

        River bluff based on math + blocker quality.
        """
        # Don't bluff with showdown value (check instead for pot control)
        if hand_strength.has_showdown_value:
            return False, 0.0

        # Don't bluff fish – they don't fold
        if player_type == "fish":
            return False, 0.0

        # Determine bluff sizing
        if player_type == "nit":
            bluff_sizing = 0.33  # small sizing enough vs nit
        else:
            bluff_sizing = 0.75  # standard river bluff

        # Need fold percentage for the bluff to be profitable
        need_fold_pct = bluff_sizing / (1.0 + bluff_sizing)  # e.g. 0.75/(1.75) = 0.43

        # Estimate villain's actual fold frequency on river
        base_fold = 0.50
        if villain_profile is not None:
            base_fold = getattr(villain_profile.stats, "fold_to_river_cbet", 0.45)

        fold_adjust = {"fish": 0.6, "nit": 1.4, "LAG": 1.0, "TAG": 1.0}
        estimated_fold = base_fold * fold_adjust.get(player_type, 1.0)
        estimated_fold = max(0.0, min(1.0, estimated_fold))

        # Calling station: skip bluff
        if estimated_fold < 0.35:
            return False, 0.0

        # Calculate blocker score using hero's actual hole cards
        # Weights: nut flush blocker > top pair/straight > sets
        _NUT_FLUSH_WEIGHT = 1.5
        _TOP_PAIR_WEIGHT = 1.0
        _STRAIGHT_WEIGHT = 1.0
        _SET_WEIGHT = 0.5
        _BLOCKER_DIVISOR = 4.0
        blocker_info = calculate_blocker_value(hero_hand, board)
        blocker_score = (
            blocker_info.get("blocks_nut_flush", 0.0) * _NUT_FLUSH_WEIGHT
            + blocker_info.get("blocks_top_pair", 0.0) * _TOP_PAIR_WEIGHT
            + blocker_info.get("blocks_straights", 0.0) * _STRAIGHT_WEIGHT
            + blocker_info.get("blocks_sets", 0.0) * _SET_WEIGHT
        ) / _BLOCKER_DIVISOR

        if estimated_fold > need_fold_pct and blocker_score >= 0.2:
            return True, bluff_sizing

        # SPR < 1: all-in bluff instead of bet
        spr = hero_stack / pot if pot > 0 else 10.0
        if spr < 1 and estimated_fold > need_fold_pct:
            return True, spr  # bet remaining stack

        return False, 0.0

    def _river_facing_bet(
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
        villain_range: List[ComboWeight],
        sizing_ratio: float,
    ) -> PostflopDecision:
        """River facing a bet: bluff-catch or fold (or raise with nuts)."""
        made = hand_strength.made_hand
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        spr = hero_stack / pot if pot > 0 else 10.0

        player_type = "TAG"
        if villain_profile is not None and villain_profile.stats.hands_played >= 30:
            player_type = villain_profile.classify()

        # Raise with the nuts
        if made >= MadeHandType.TRIPS_SET:
            if spr < 3:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"River all-in: {made.name} (nut/near-nut hand)",
                    confidence=0.95,
                )
            raise_size = min(hero_stack, pot * 0.85 + to_call * 2.5)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"River raise for value: {made.name}",
                confidence=0.90,
            )

        # No showdown value → fold (never bluff-catch with air)
        if not hand_strength.has_showdown_value:
            return PostflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"River fold: {made.name} has no showdown value",
                confidence=0.80,
            )

        # Villain-type shortcuts before bluff-catch math
        if player_type == "fish":
            # Fish bet = real hand; fold marginals
            if made < MadeHandType.TOP_PAIR_TOP_KICKER and sizing_ratio > 0.5:
                return PostflopDecision(
                    action="fold",
                    amount=0.0,
                    reasoning=f"River fold vs fish large bet: {made.name} (fish bet = value)",
                    confidence=0.72,
                )
        elif player_type == "nit":
            # Nit almost never bluffs; fold unless monster
            if made < MadeHandType.TRIPS_SET:
                return PostflopDecision(
                    action="fold",
                    amount=0.0,
                    reasoning=f"River fold vs nit bet: {made.name} (nit rarely bluffs)",
                    confidence=0.78,
                )

        # Bluff-catch math
        should_call, reasoning = self._river_bluff_catch(
            hand_strength, villain_profile, player_type, villain_range,
            board, pot, to_call, sizing_ratio,
        )
        if should_call:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=reasoning,
                confidence=0.60,
            )

        # Default fold
        if hand_strength.has_showdown_value and made >= MadeHandType.TOP_PAIR_GOOD_KICKER:
            # Marginal call if pot odds are close
            if pot_odds < 0.35:
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"River marginal call: {made.name}, pot odds {pot_odds:.0%}",
                    confidence=0.52,
                )

        return PostflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"River fold: {made.name}, bluff-catch not profitable (pot_odds={pot_odds:.0%})",
            confidence=0.65,
        )

    def _river_bluff_catch(
        self,
        hand_strength: HandStrength,
        villain_profile: Optional[VillainProfile],
        player_type: str,
        villain_range: List[ComboWeight],
        board: List[Card],
        pot: float,
        to_call: float,
        sizing_ratio: float,
    ) -> Tuple[bool, str]:
        """Return (should_call, reasoning) for river bluff-catch decision.

        Estimates villain's bluff frequency based on action line patterns
        and player type, then compares to pot odds.
        """
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5

        # Base bluff estimate: assume moderate triple-barrel range
        # (No explicit action history here; use heuristic from sizing)
        if sizing_ratio > 1.0:
            # Overbet: polarized – some bluffs exist
            base_bluff_pct = 0.35
        elif sizing_ratio > 0.6:
            # Standard large bet
            base_bluff_pct = 0.30
        else:
            # Small bet: usually thin value; fewer bluffs
            base_bluff_pct = 0.18

        # Adjust by player type
        type_multipliers = {
            "fish": 0.3,   # fish almost never bluff
            "nit": 0.2,    # nit never bluffs
            "LAG": 1.5,    # LAG bluffs a lot
            "TAG": 1.0,    # standard
        }
        estimated_bluff_pct = base_bluff_pct * type_multipliers.get(player_type, 1.0)
        estimated_bluff_pct = max(0.0, min(1.0, estimated_bluff_pct))

        should_call = estimated_bluff_pct >= pot_odds
        reasoning = (
            f"River bluff-catch: estimated_bluff={estimated_bluff_pct:.0%} "
            f"vs pot_odds={pot_odds:.0%} ({player_type})"
        )
        return should_call, reasoning

    # ------------------------------------------------------------------
    # Turn independent decision framework (Improvement 1)
    # ------------------------------------------------------------------

    def _analyze_turn_change(
        self,
        board: List[Card],
        turn_card: Optional[Card],
        hero_hand: Tuple[Card, Card],
    ) -> TurnChange:
        """Analyze how the turn card affects range dynamics."""
        if turn_card is None or len(board) < 4:
            return TurnChange(
                is_blank=True, is_overcard=False, completes_flush=False,
                completes_straight=False, pairs_board=False, is_scare_card=False,
                range_shift="neutral",
            )

        flop = board[:3]
        tags = analyze_runout(flop, turn_card)

        is_blank = "blank" in tags
        is_overcard = "overcard" in tags
        completes_flush = "flush_completes" in tags
        completes_straight = "straight_completes" in tags
        pairs_board = "paired_board" in tags
        is_scare_card = "scary_card" in tags

        # Determine range shift based on what happened
        hero_draw = classify_hand(hero_hand, flop).draw
        hero_full = classify_hand(hero_hand, board)

        range_shift = "neutral"

        if completes_flush:
            # Hero benefits if they have the flush, else villain is more dangerous
            if hero_full.made_hand >= MadeHandType.FLUSH_LOW:
                range_shift = "hero_better"
            else:
                range_shift = "villain_better"
        elif completes_straight:
            if hero_full.made_hand >= MadeHandType.STRAIGHT_NON_NUT:
                range_shift = "hero_better"
            else:
                range_shift = "villain_better"
        elif pairs_board:
            # Paired board usually helps villain (set → full house) unless hero has set
            if hero_full.made_hand >= MadeHandType.TRIPS_SET:
                range_shift = "hero_better"
            else:
                range_shift = "villain_better"
        elif is_blank:
            range_shift = "neutral"
        elif is_scare_card or is_overcard:
            # A/K on low board: depends on position (caller tends to have A/K)
            range_shift = "villain_better"

        return TurnChange(
            is_blank=is_blank,
            is_overcard=is_overcard,
            completes_flush=completes_flush,
            completes_straight=completes_straight,
            pairs_board=pairs_board,
            is_scare_card=is_scare_card,
            range_shift=range_shift,
        )

    def _turn_first_to_act(
        self,
        hero_hand: Tuple[Card, Card],
        board: List[Card],
        pot: float,
        hero_stack: float,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        position: Position,
        villain_profile: Optional[VillainProfile],
        barrel_plan: Optional[BarrelPlan],
        is_multiway: bool,
        multiway_adj,
        villain_stack: float,
        turn_change: TurnChange,
    ) -> PostflopDecision:
        """Turn first-to-act with runout awareness."""
        spr = hero_stack / pot if pot > 0 else 10.0
        pos_label = "ip" if position in (Position.BTN, Position.CO, Position.HJ) else "oop"
        in_position = pos_label == "ip"
        made = hand_strength.made_hand
        draw = hand_strength.draw

        # Check-raise evaluation (OOP, strong hands)
        if not in_position:
            cr_decision = self._evaluate_check_raise(
                hand_strength, board_texture, range_adv,
                spr, pot, villain_profile, "turn", is_multiway,
            )
            if cr_decision is not None:
                return cr_decision

        # Donk/probe bet evaluation (OOP, Improvement 2)
        if not in_position:
            donk_decision = self._evaluate_donk_or_probe(
                hand_strength, board_texture, range_adv, "turn",
                pot, hero_stack, villain_profile, [], is_multiway, board,
            )
            if donk_decision is not None:
                return donk_decision

        # ------------------------------------------------------------------
        # Runout-aware turn adjustments
        # ------------------------------------------------------------------

        # Flush/straight completed and hero doesn't have it → stop bluffing
        if (turn_change.completes_flush or turn_change.completes_straight) and turn_change.range_shift == "villain_better":
            if not hand_strength.has_showdown_value or made < MadeHandType.TOP_PAIR_GOOD_KICKER:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn check: draw completed, hero doesn't have it ({made.name})",
                    confidence=0.70,
                )

        # Scare card (A/K) → slow down unless hero has strong hand
        if turn_change.is_scare_card and made < MadeHandType.TOP_PAIR_TOP_KICKER:
            return PostflopDecision(
                action="check",
                amount=0.0,
                reasoning=f"Turn check: scare card, slow down with {made.name}",
                confidence=0.68,
            )

        # Board paired and hero doesn't have boat → slow down
        if turn_change.pairs_board and made < MadeHandType.FULL_HOUSE and draw == DrawType.NONE:
            if made < MadeHandType.TOP_PAIR_TOP_KICKER:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn check: board paired, careful with {made.name}",
                    confidence=0.65,
                )

        # Villain capped (flop check-call, turn check) → can apply pressure
        # This is handled implicitly through barrel plan; overbet opportunity
        flop_check_call_turn_check = (
            barrel_plan is not None
            and barrel_plan.flop_action == "bet"
            and not in_position
        )

        # Follow barrel plan if available
        if barrel_plan is not None:
            turn_card = board[3] if len(board) >= 4 else None
            planned_action = get_current_action(
                barrel_plan, "turn",
                turn_card=turn_card,
                board=board,
            )
            if planned_action in ("bet", "bluff", "barrel"):
                if self._should_value_bet(hand_strength, board_texture, range_adv, position, spr):
                    sizing = calculate_sizing(
                        hand_strength, board_texture, range_adv, "turn",
                        spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
                    )
                    bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                    return PostflopDecision(
                        action="bet" if bet_amount < hero_stack else "all-in",
                        amount=bet_amount,
                        reasoning=f"Turn value bet ({sizing.reasoning}): {made.name}",
                        confidence=0.80,
                    )
                if self._should_bluff(hand_strength, board_texture, range_adv, position, spr, pot):
                    sizing = calculate_sizing(
                        hand_strength, board_texture, range_adv, "turn",
                        spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
                    )
                    bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Turn bluff ({sizing.reasoning}): {draw.name}",
                        confidence=0.55,
                    )
            if planned_action in ("check", "give_up"):
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn: following barrel plan ({planned_action})",
                    confidence=0.65,
                )

        # No plan: decide from scratch
        if self._should_value_bet(hand_strength, board_texture, range_adv, position, spr):
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, "turn",
                spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
            )
            if is_multiway and not multiway_adj.bluff_allowed:
                if made < MadeHandType.TOP_PAIR_TOP_KICKER + multiway_adj.value_threshold_adjustment:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning="Turn multiway: tightening value range",
                        confidence=0.65,
                    )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot * multiway_adj.bet_frequency_multiplier)
            if bet_amount < pot * 0.20:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning="Turn bet too small after multiway adjustment, check",
                    confidence=0.60,
                )
            return PostflopDecision(
                action="bet" if bet_amount < hero_stack else "all-in",
                amount=bet_amount,
                reasoning=f"Turn value bet ({sizing.reasoning}): {made.name}",
                confidence=0.78,
            )

        if self._should_bluff(hand_strength, board_texture, range_adv, position, spr, pot):
            if is_multiway and not multiway_adj.bluff_allowed:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning="Turn multiway: no bluffing",
                    confidence=0.70,
                )
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, "turn",
                spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
            )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Turn bluff ({sizing.reasoning}): {draw.name}",
                confidence=0.50,
            )

        return PostflopDecision(
            action="check",
            amount=0.0,
            reasoning=f"Turn check: {made.name} – no clear action",
            confidence=0.65,
        )

    def _turn_facing_bet(
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
        villain_range: List[ComboWeight],
        sizing_ratio: float,
        turn_change: TurnChange,
        barrel_plan: Optional[BarrelPlan],
        villain_stack: float,
    ) -> PostflopDecision:
        """Turn facing a bet: double barrel defense with runout awareness."""
        made = hand_strength.made_hand
        draw = hand_strength.draw
        spr = hero_stack / pot if pot > 0 else 10.0
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        equity = hand_strength.equity_bucket

        # Overbet on turn (>80% pot) → treat as polarized
        is_overbet = sizing_ratio > 0.80
        if is_overbet:
            if made >= MadeHandType.TRIPS_SET:
                if spr < 2:
                    return PostflopDecision(
                        action="all-in",
                        amount=hero_stack,
                        reasoning=f"Turn overbet: all-in with {made.name} (low SPR)",
                        confidence=0.92,
                    )
                raise_size = min(hero_stack, pot * 0.85 + to_call * 2.5)
                return PostflopDecision(
                    action="raise",
                    amount=raise_size,
                    reasoning=f"Turn overbet: raising with {made.name} (polarized spot)",
                    confidence=0.82,
                )
            if made >= MadeHandType.TOP_TWO_PAIR:
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Turn overbet call: {made.name} vs polarized sizing",
                    confidence=0.65,
                )
            # Overbet with marginal hand → fold unless deep with draw
            if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
                # Turn draw equity: 1 card to come (more precise)
                # OESD ≈ 17.4%, flush draw ≈ 19.6%
                if draw == DrawType.FLUSH_DRAW_NUT:
                    effective_equity = max(equity, 0.196)
                elif draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
                    effective_equity = max(equity, 0.35)
                elif draw == DrawType.OESD:
                    effective_equity = max(equity, 0.174)
                else:
                    effective_equity = equity
                if effective_equity > pot_odds and spr > 5:
                    return PostflopDecision(
                        action="call",
                        amount=to_call,
                        reasoning=f"Turn overbet: implied odds call with {draw.name} (equity={effective_equity:.0%})",
                        confidence=0.55,
                    )
            return PostflopDecision(
                action="fold",
                amount=0.0,
                reasoning=f"Turn overbet fold: {made.name} vs polarized overbet",
                confidence=0.68,
            )

        # Double barrel defense: need stronger hand vs two streets of aggression
        # Use 1-card-to-come equity for draws
        if draw == DrawType.FLUSH_DRAW_NUT:
            effective_equity = max(equity, 0.196)
        elif draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
            effective_equity = max(equity, 0.35)
        elif draw == DrawType.OESD:
            effective_equity = max(equity, 0.174)
        elif draw == DrawType.FLUSH_DRAW_LOW:
            effective_equity = max(equity, 0.196)
        elif draw == DrawType.GUTSHOT:
            effective_equity = max(equity, 0.087)
        else:
            effective_equity = equity

        # Monster: raise or call
        if made >= MadeHandType.TRIPS_SET:
            if spr < 3:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"Turn: all-in with {made.name} (low SPR)",
                    confidence=0.95,
                )
            raise_size = min(hero_stack, pot * 0.75 + to_call * 2.5)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"Turn: raise for value with {made.name}",
                confidence=0.88,
            )

        # Strong value: call
        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            if effective_equity > pot_odds:
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Turn: call with {made.name} (equity={effective_equity:.0%} > pot_odds={pot_odds:.0%})",
                    confidence=0.78,
                )

        # Turn completed a flush/straight and we don't have it → tighten call range
        if (turn_change.completes_flush or turn_change.completes_straight) and turn_change.range_shift == "villain_better":
            if made < MadeHandType.TOP_TWO_PAIR and draw == DrawType.NONE:
                return PostflopDecision(
                    action="fold",
                    amount=0.0,
                    reasoning=f"Turn fold: draw completed, {made.name} not strong enough",
                    confidence=0.72,
                )

        # Nut draws: call or semi-raise
        if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT):
            if effective_equity > pot_odds + 0.05:
                if spr > 2 and not is_multiway:
                    raise_size = min(hero_stack, pot * 0.85 + to_call * 2)
                    return PostflopDecision(
                        action="raise",
                        amount=raise_size,
                        reasoning=f"Turn semi-raise: {draw.name} (equity={effective_equity:.0%})",
                        confidence=0.68,
                    )
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"Turn semi-call: {draw.name} (equity={effective_equity:.0%})",
                    confidence=0.62,
                )

        # Implied odds: deep stack (SPR > 5) → call with draws
        if effective_equity > pot_odds and draw != DrawType.NONE and spr > 5:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"Turn implied odds call: {draw.name} (SPR={spr:.1f}, equity={effective_equity:.0%})",
                confidence=0.58,
            )

        # Positive EV call
        ev_call = self._calculate_call_ev(to_call, pot, effective_equity)
        if ev_call > 0 and hand_strength.has_showdown_value:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"Turn call: positive EV ({ev_call:.2f}), {made.name}",
                confidence=0.58,
            )

        return PostflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"Turn fold: equity={effective_equity:.0%} < pot_odds={pot_odds:.0%}, {made.name}",
            confidence=0.68,
        )

    # ------------------------------------------------------------------
    # Donk bet / probe bet (Improvement 2)
    # ------------------------------------------------------------------

    def _villain_checked_back_flop(self, action_history: List[Dict]) -> bool:
        """Return True if the villain checked back on the flop (no flop cbet).

        Checks the action history for a sequence indicating the villain passed
        on a flop continuation bet opportunity.
        """
        if not action_history:
            return False
        flop_actions = [a for a in action_history if a.get("street") == "flop"]
        if not flop_actions:
            return False
        # Villain checked back: the last flop action was a check by villain
        for action in flop_actions:
            if action.get("actor") == "villain" and action.get("action") == "check":
                return True
        return False

    def _evaluate_donk_or_probe(
        self,
        hand_strength: HandStrength,
        board_texture,
        range_adv: RangeAdvantage,
        street: str,
        pot: float,
        hero_stack: float,
        villain_profile: Optional[VillainProfile],
        action_history: List[Dict],
        is_multiway: bool,
        board: List[Card],
    ) -> Optional[PostflopDecision]:
        """Evaluate donk bet (flop OOP) or probe bet (turn after villain check-back).

        Returns a PostflopDecision if we should lead-out, or None to fall
        through to the normal barrel plan logic.
        """
        if is_multiway:
            return None

        made = hand_strength.made_hand
        draw = hand_strength.draw
        spr = hero_stack / pot if pot > 0 else 10.0

        # ── Donk Bet (Flop, OOP) ──
        if street == "flop":
            # Don't donk vs villain who cbets > 75% (let him cbet, then check-raise)
            if villain_profile is not None:
                cbet_freq = getattr(villain_profile.stats, "cbet_flop", 0.6)
                if cbet_freq > 0.75:
                    return None

            # Need at least some hand or draw
            has_value = made >= MadeHandType.MIDDLE_PAIR or draw != DrawType.NONE
            if not has_value:
                return None

            board_cards = board[:3] if len(board) >= 3 else board
            if not board_cards:
                return None
            board_has_ace = any(c.rank == Rank.ACE for c in board_cards)
            board_has_top_card = board_cards and max(c.rank for c in board_cards) >= Rank.KING

            # Range advantage donk: board has A and we're from BB → BB has more Ax
            if board_has_ace and made >= MadeHandType.TOP_PAIR_WEAK_KICKER:
                bet_fraction = 0.30  # 25-33% pot
                bet_amount = min(hero_stack, pot * bet_fraction)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Donk bet: range advantage on A-high board, {made.name} ({bet_fraction:.0%} pot)",
                    confidence=0.62,
                )

            # Protection donk: vulnerable hand + wet board
            if (
                hand_strength.is_vulnerable
                and made >= MadeHandType.MIDDLE_PAIR
                and board_texture.wetness >= 6
            ):
                bet_fraction = 0.40  # 33-50% pot
                bet_amount = min(hero_stack, pot * bet_fraction)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Donk bet: protection on wet board, {made.name} ({bet_fraction:.0%} pot)",
                    confidence=0.60,
                )

            # Value+protection donk: connected board + two pair
            if made >= MadeHandType.BOTTOM_TWO_PAIR and board_texture.connectivity >= 6:
                bet_fraction = 0.45  # 40-50% pot
                bet_amount = min(hero_stack, pot * bet_fraction)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Donk bet: value+protection on connected board, {made.name} ({bet_fraction:.0%} pot)",
                    confidence=0.65,
                )

            return None

        # ── Probe Bet (Turn, after villain flop check-back) ──
        if street == "turn":
            # SPR too low → just bet for value (let normal logic handle)
            if spr < 1.5:
                return None

            villain_checked_flop = self._villain_checked_back_flop(action_history)
            if not villain_checked_flop:
                return None

            # No showdown value and no draw → don't probe with air
            if not hand_strength.has_showdown_value and draw == DrawType.NONE:
                return None

            # Value probe: middle pair+ vs capped range
            if made >= MadeHandType.MIDDLE_PAIR and hand_strength.has_showdown_value:
                bet_fraction = 0.45  # 40-50% pot
                bet_amount = min(hero_stack, pot * bet_fraction)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Probe bet: value vs capped range, {made.name} ({bet_fraction:.0%} pot)",
                    confidence=0.63,
                )

            # Bluff probe: draw + villain folds enough
            if draw != DrawType.NONE:
                fold_freq = 0.45  # default
                if villain_profile is not None:
                    fold_freq = getattr(villain_profile.stats, "fold_to_turn_cbet", 0.45)
                if fold_freq > 0.50:
                    bet_fraction = 0.36  # 33-40% pot
                    bet_amount = min(hero_stack, pot * bet_fraction)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Probe bluff: {draw.name} vs capped range (fold_to_turn_cbet={fold_freq:.0%})",
                        confidence=0.57,
                    )

            # Merge probe: weak TP / middle pair, deny free card
            if made >= MadeHandType.TOP_PAIR_WEAK_KICKER:
                bet_fraction = 0.40  # 33-50% pot
                bet_amount = min(hero_stack, pot * bet_fraction)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Probe bet: merge/deny free card, {made.name} ({bet_fraction:.0%} pot)",
                    confidence=0.58,
                )

            return None

        return None
