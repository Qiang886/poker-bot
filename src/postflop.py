"""Post-flop decision engine."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.card import Card, Rank
from src.position import Position, is_in_position
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand
from src.board_analysis import analyze_board, analyze_range_advantage, RangeAdvantage, NutAdvantage
from src.sizing import calculate_sizing, SizingProfile
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
    """Describes what the turn card brought relative to the flop."""
    is_blank: bool           # 2-6 and no draw completed
    is_overcard: bool        # higher than all flop cards
    completes_flush: bool    # 3rd card of same suit
    completes_straight: bool # completes a straight draw
    pairs_board: bool        # pairs a flop card
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
        action_history: Optional[List[Dict]] = None,
    ) -> PostflopDecision:
        """Top-level postflop decision entry-point."""
        hand_strength = classify_hand(hero_hand, board)
        board_texture = analyze_board(board)
        range_adv = analyze_range_advantage(hero_range, villain_range, board)

        # Determine position label
        pos_label = "ip" if position in (Position.BTN, Position.CO) else "oop"

        spr = hero_stack / pot if pot > 0 else 10.0
        multiway_adj = adjust_for_multiway(hand_strength, 2 if not is_multiway else 3, board_texture)

        # Dynamic equity adjustment
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

        if action_history is None:
            action_history = []

        if street == "turn":
            turn_card = board[3] if len(board) >= 4 else None
            turn_change = self._analyze_turn_change(board, turn_card, hero_hand)
            if to_call > 0:
                actual_ratio = sizing_ratio if sizing_ratio > 0 else (to_call / pot if pot > 0 else 0.5)
                base_decision = self._turn_facing_bet(
                    hero_hand, board, pot, to_call, hero_stack,
                    hand_strength, board_texture, range_adv, position, villain_profile,
                    is_multiway, turn_change, villain_range, actual_ratio, barrel_plan,
                )
            else:
                base_decision = self._turn_first_to_act(
                    hero_hand, board, pot, hero_stack,
                    hand_strength, board_texture, range_adv, position, villain_profile,
                    barrel_plan, is_multiway, multiway_adj, villain_stack,
                    turn_change, action_history,
                )
        elif to_call > 0:
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
                action_history=action_history,
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
        action_history: Optional[List[Dict]] = None,
    ) -> PostflopDecision:
        spr = hero_stack / pot if pot > 0 else 10.0
        pos_label = "ip" if position in (Position.BTN, Position.CO, Position.HJ) else "oop"
        in_position = pos_label == "ip"

        if action_history is None:
            action_history = []

        # Check-raise evaluation (OOP only, before barrel plan)
        if not in_position:
            cr_decision = self._evaluate_check_raise(
                hand_strength, board_texture, range_adv,
                spr, pot, villain_profile, street, is_multiway,
            )
            if cr_decision is not None:
                return cr_decision

        # Donk bet / Probe bet evaluation (OOP only, flop/turn)
        if not in_position and street in ("flop", "turn"):
            donk_decision = self._evaluate_donk_or_probe(
                hand_strength, board_texture, range_adv, street,
                pot, hero_stack, villain_profile, action_history,
                is_multiway, board,
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

    # ------------------------------------------------------------------
    # Turn-specific analysis
    # ------------------------------------------------------------------

    def _analyze_turn_change(
        self,
        board: List[Card],
        turn_card: Optional[Card],
        hero_hand: Tuple[Card, Card],
    ) -> TurnChange:
        """Analyze what the turn card brings relative to the flop."""
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

        # Determine range shift
        hero_has_flush = any(c.suit == turn_card.suit for c in hero_hand) and completes_flush
        hero_has_straight = completes_straight  # simplified: assume hero can have it

        range_shift = "neutral"
        if is_blank:
            range_shift = "neutral"
        elif completes_flush:
            range_shift = "hero_better" if hero_has_flush else "villain_better"
        elif completes_straight:
            range_shift = "neutral"
        elif is_scare_card:
            # A/K on low board: villain (preflop raiser) has more Ax/Kx
            range_shift = "villain_better"
        elif pairs_board:
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
        action_history: Optional[List[Dict]] = None,
    ) -> PostflopDecision:
        """Turn-specific first-to-act decision with runout awareness."""
        spr = hero_stack / pot if pot > 0 else 10.0
        pos_label = "ip" if position in (Position.BTN, Position.CO, Position.HJ) else "oop"
        in_position = pos_label == "ip"
        made = hand_strength.made_hand
        draw = hand_strength.draw
        equity = hand_strength.equity_bucket

        if action_history is None:
            action_history = []

        # Check-raise evaluation OOP (turn check-raise with monster)
        if not in_position:
            cr_decision = self._evaluate_check_raise(
                hand_strength, board_texture, range_adv,
                spr, pot, villain_profile, "turn", is_multiway,
            )
            if cr_decision is not None:
                return cr_decision

        # Probe bet evaluation OOP
        if not in_position:
            probe_decision = self._evaluate_donk_or_probe(
                hand_strength, board_texture, range_adv, "turn",
                pot, hero_stack, villain_profile, action_history,
                is_multiway, board,
            )
            if probe_decision is not None:
                return probe_decision

        # Adjust barrel plan based on turn change
        if barrel_plan is not None:
            turn_card = board[3] if len(board) >= 4 else None
            planned_action = get_current_action(
                barrel_plan, "turn",
                turn_card=turn_card,
                board=board,
            )

            # Override: scare card or flush complete with no hero flush → slow down
            if turn_change.is_scare_card or (turn_change.completes_flush and not any(
                c.suit == board[3].suit for c in hero_hand
            )):
                if made < MadeHandType.TRIPS_SET:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Turn slow down: {('scare card' if turn_change.is_scare_card else 'flush completes')} | {made.name}",
                        confidence=0.72,
                    )

            # Override: board pairs and hero has no boat
            if turn_change.pairs_board and made < MadeHandType.FULL_HOUSE:
                if made < MadeHandType.TRIPS_SET:
                    return PostflopDecision(
                        action="check",
                        amount=0.0,
                        reasoning=f"Turn slow down: board pairs, no boat | {made.name}",
                        confidence=0.68,
                    )

            if planned_action in ("bet", "bluff", "barrel"):
                # Draw completed to value hand: upgrade from semi-bluff to value bet
                if draw != DrawType.NONE and (turn_change.completes_flush or turn_change.completes_straight):
                    new_made = classify_hand(hero_hand, board).made_hand
                    if new_made >= MadeHandType.FLUSH_LOW:
                        sizing = calculate_sizing(
                            hand_strength, board_texture, range_adv, "turn",
                            spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
                        )
                        bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                        return PostflopDecision(
                            action="bet" if bet_amount < hero_stack else "all-in",
                            amount=bet_amount,
                            reasoning=f"Turn: draw completed to {new_made.name}, value bet",
                            confidence=0.85,
                        )

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
                    # Reduce bluff frequency on scary turns
                    if turn_change.range_shift == "villain_better" and not turn_change.is_blank:
                        return PostflopDecision(
                            action="check",
                            amount=0.0,
                            reasoning=f"Turn: range shifted villain_better, no bluff | {draw.name}",
                            confidence=0.65,
                        )
                    sizing = calculate_sizing(
                        hand_strength, board_texture, range_adv, "turn",
                        spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
                    )
                    bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
                    return PostflopDecision(
                        action="bet",
                        amount=bet_amount,
                        reasoning=f"Turn semi-bluff ({sizing.reasoning}): {draw.name}",
                        confidence=0.55,
                    )
            if planned_action in ("check", "give_up"):
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn: following barrel plan: {planned_action}",
                    confidence=0.65,
                )

        # Villain capped (check-check on flop = no monster): can overbet/thin value
        villain_capped = self._villain_checked_back_flop(action_history)
        if villain_capped and equity > 0.55 and in_position:
            # Opponent flop check-back → range capped → can bet wider
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, "turn",
                spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
            )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
            if bet_amount >= pot * 0.25:
                return PostflopDecision(
                    action="bet" if bet_amount < hero_stack else "all-in",
                    amount=bet_amount,
                    reasoning=f"Turn: villain capped (flop check-back), thin value | {made.name}",
                    confidence=0.72,
                )

        # No plan: fall back to standard logic with turn adjustments
        if self._should_value_bet(hand_strength, board_texture, range_adv, position, spr):
            # Scare card: tighten value range on turn
            if turn_change.is_scare_card and made < MadeHandType.OVERPAIR_BIG:
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn: scare card, tighten value range | {made.name}",
                    confidence=0.70,
                )
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, "turn",
                spr, pot, hero_stack, is_value=True, is_multiway=is_multiway,
            )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
            return PostflopDecision(
                action="bet" if bet_amount < hero_stack else "all-in",
                amount=bet_amount,
                reasoning=f"Turn value bet ({sizing.reasoning}): {made.name}",
                confidence=0.78,
            )

        if self._should_bluff(hand_strength, board_texture, range_adv, position, spr, pot):
            if turn_change.range_shift == "villain_better":
                return PostflopDecision(
                    action="check",
                    amount=0.0,
                    reasoning=f"Turn: villain_better range shift, skip bluff | {draw.name}",
                    confidence=0.65,
                )
            sizing = calculate_sizing(
                hand_strength, board_texture, range_adv, "turn",
                spr, pot, hero_stack, is_value=False, is_multiway=is_multiway,
            )
            bet_amount = min(hero_stack, pot * sizing.fraction_of_pot)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Turn semi-bluff ({sizing.reasoning}): {draw.name}",
                confidence=0.50,
            )

        return PostflopDecision(
            action="check",
            amount=0.0,
            reasoning=f"Turn check: {made.name} – no clear bet",
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
        turn_change: TurnChange,
        villain_range: List[ComboWeight],
        sizing_ratio: float,
        barrel_plan: Optional[BarrelPlan],
    ) -> PostflopDecision:
        """Turn-specific facing-bet decision with double-barrel awareness."""
        spr = hero_stack / pot if pot > 0 else 10.0
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
        equity = hand_strength.equity_bucket
        made = hand_strength.made_hand
        draw = hand_strength.draw

        sizing_interp = self._sizing_tell.interpret(
            action="bet",
            sizing_ratio=sizing_ratio,
            street="turn",
            board_texture=board_texture,
            villain_profile=villain_profile,
        )
        sizing_note = f"[turn sizing: {sizing_interp.polarization}] "

        # Outs analysis for draws on turn (only 1 card to come: lower equity)
        outs_note = ""
        outs_analysis = None
        if draw != DrawType.NONE and len(board) >= 4:
            outs_analysis = self._outs_calc.calculate_outs(
                hero_hand, board, villain_range, simulations_per_out=150
            )
            if outs_analysis.total_clean > 0:
                outs_note = f" | outs: {format_outs_summary(outs_analysis)}"
                if outs_analysis.true_equity > equity:
                    equity = outs_analysis.true_equity

        # Turn equity floors (lower than flop: only 1 card to come)
        if draw == DrawType.FLUSH_DRAW_NUT:
            equity = max(equity, 0.20)   # ~9/46 ≈ 19.6%
        elif draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
            equity = max(equity, 0.35)
        elif draw == DrawType.OESD:
            equity = max(equity, 0.17)   # ~8/46 ≈ 17.4%
        elif draw == DrawType.FLUSH_DRAW_LOW:
            equity = max(equity, 0.18)
        elif draw == DrawType.GUTSHOT:
            equity = max(equity, 0.09)

        ev_call = self._calculate_call_ev(to_call, pot, equity)

        # Turn overbet (>80% pot) = polarized: call strong, fold medium
        is_overbet = sizing_ratio > 0.80

        # Monster: raise for value
        if made >= MadeHandType.TRIPS_SET:
            if spr < 3:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"{sizing_note}{made.name}: turn low SPR shove",
                    confidence=0.95,
                )
            raise_size = min(hero_stack, pot * 0.75 + to_call * 2.5)
            return PostflopDecision(
                action="raise",
                amount=raise_size,
                reasoning=f"{sizing_note}{made.name}: turn raise for value",
                confidence=0.90,
            )

        # Strong value: call (raise if overbet and strong)
        if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
            if spr < 2:
                return PostflopDecision(
                    action="all-in",
                    amount=hero_stack,
                    reasoning=f"{sizing_note}{made.name}: turn commit stack",
                    confidence=0.85,
                )
            if ev_call > 0:
                return PostflopDecision(
                    action="call",
                    amount=to_call,
                    reasoning=f"{sizing_note}{made.name}: turn call (equity={equity:.0%}, pot_odds={pot_odds:.0%})",
                    confidence=0.78,
                )

        # Double-barrel defense: need stronger hand vs aggressive double-barrel
        # (villain bet flop AND turn = narrower range)
        is_double_barrel = self._is_double_barrel(villain_profile, sizing_ratio)
        double_barrel_threshold = MadeHandType.TOP_PAIR_GOOD_KICKER if not is_double_barrel else MadeHandType.TOP_PAIR_TOP_KICKER

        # Flush/straight complete changes calling requirements
        if turn_change.completes_flush or turn_change.completes_straight:
            # Hero has no flush/straight and didn't complete → raise bar
            hero_has_made = made >= MadeHandType.FLUSH_LOW
            if not hero_has_made and made < MadeHandType.TRIPS_SET:
                if made < MadeHandType.TOP_PAIR_TOP_KICKER:
                    return PostflopDecision(
                        action="fold",
                        amount=0.0,
                        reasoning=f"{sizing_note}Turn: draw completed, fold {made.name}",
                        confidence=0.70,
                    )

        # Nut draws with implied odds
        if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.COMBO_DRAW_NUT):
            if spr > 5:
                # Deep stack: implied odds justify call
                if equity > pot_odds * 0.85:
                    return PostflopDecision(
                        action="call",
                        amount=to_call,
                        reasoning=f"{sizing_note}Turn draw call (implied odds, SPR={spr:.1f}): {draw.name}{outs_note}",
                        confidence=0.62,
                    )
            else:
                # Short stack: pure pot odds
                if equity > pot_odds:
                    return PostflopDecision(
                        action="call",
                        amount=to_call,
                        reasoning=f"{sizing_note}Turn draw call (pot odds): {draw.name}{outs_note}",
                        confidence=0.58,
                    )

        # Positive EV call for medium hands
        if ev_call > 0 and hand_strength.has_showdown_value:
            # With overbet, be more selective
            if is_overbet and made < MadeHandType.TOP_PAIR_TOP_KICKER:
                return PostflopDecision(
                    action="fold",
                    amount=0.0,
                    reasoning=f"{sizing_note}Turn: overbet vs {made.name}, fold",
                    confidence=0.65,
                )
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"{sizing_note}Turn call: positive EV, {made.name}",
                confidence=0.58,
            )

        # Draw with pot odds
        if equity > pot_odds and draw != DrawType.NONE:
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"{sizing_note}Turn draw call: equity {equity:.0%} > pot odds {pot_odds:.0%}{outs_note}",
                confidence=0.52,
            )

        # Bluff-catch vs polarized (overbet on turn = bluff or nut)
        if (
            sizing_interp.polarization in ("polarized", "very_polarized")
            and hand_strength.has_showdown_value
            and equity >= pot_odds * 0.90
            and made >= MadeHandType.TOP_PAIR_GOOD_KICKER
        ):
            return PostflopDecision(
                action="call",
                amount=to_call,
                reasoning=f"{sizing_note}Turn bluff-catch: {made.name} vs polarized bet",
                confidence=0.52,
            )

        return PostflopDecision(
            action="fold",
            amount=0.0,
            reasoning=f"{sizing_note}Turn fold: equity {equity:.0%} < pot odds {pot_odds:.0%}, {made.name}",
            confidence=0.70,
        )

    def _is_double_barrel(self, villain_profile: Optional[VillainProfile], turn_sizing_ratio: float) -> bool:
        """Heuristic: detect if villain is likely double-barreling."""
        if villain_profile is None:
            return turn_sizing_ratio > 0.40
        # High cbet_flop + large turn sizing = likely double barrel
        cbet_flop = getattr(villain_profile.stats, "cbet_flop", 0.6)
        return cbet_flop > 0.55 and turn_sizing_ratio > 0.40

    # ------------------------------------------------------------------
    # Donk bet / Probe bet
    # ------------------------------------------------------------------

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
        """
        Evaluate whether to donk bet (flop) or probe bet (turn after villain check-back).
        Returns a PostflopDecision to bet, or None to continue with normal logic.
        """
        if is_multiway:
            return None

        spr = hero_stack / pot if pot > 0 else 10.0
        if spr < 1.5:
            return None

        made = hand_strength.made_hand
        draw = hand_strength.draw
        equity = hand_strength.equity_bucket

        # Villain cbets very frequently: let them cbet, check-raise instead of donking
        cbet_freq = 0.60
        if villain_profile is not None:
            cbet_freq = getattr(villain_profile.stats, "cbet_flop", 0.60)
        if cbet_freq > 0.75 and street == "flop":
            return None

        if street == "flop":
            return self._evaluate_donk_bet(
                hand_strength, board_texture, board, pot, hero_stack,
                villain_profile, cbet_freq,
            )
        elif street == "turn":
            # Probe bet requires villain to have checked back flop
            if not self._villain_checked_back_flop(action_history):
                return None
            return self._evaluate_probe_bet(
                hand_strength, board_texture, pot, hero_stack, villain_profile, spr,
            )
        return None

    def _evaluate_donk_bet(
        self,
        hand_strength: HandStrength,
        board_texture,
        board: List[Card],
        pot: float,
        hero_stack: float,
        villain_profile: Optional[VillainProfile],
        cbet_freq: float,
    ) -> Optional[PostflopDecision]:
        """Evaluate flop donk bet conditions."""
        made = hand_strength.made_hand
        equity = hand_strength.equity_bucket

        if made < MadeHandType.BOTTOM_PAIR and hand_strength.draw == DrawType.NONE:
            return None

        # a. Range advantage donk: A-high board and we have Ax
        if board and board_texture.highest_card == Rank.ACE:
            if made >= MadeHandType.TOP_PAIR_WEAK_KICKER:
                bet_amount = min(hero_stack, pot * 0.30)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Donk bet: range advantage on A-high board | {made.name}",
                    confidence=0.62,
                )

        # b. Protection donk: vulnerable middle pair on wet board
        if (
            made in (MadeHandType.MIDDLE_PAIR, MadeHandType.TOP_PAIR_WEAK_KICKER)
            and hand_strength.is_vulnerable
            and board_texture.wetness >= 6
        ):
            bet_amount = min(hero_stack, pot * 0.40)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Donk bet: protection with vulnerable {made.name} on wet board",
                confidence=0.58,
            )

        # c. Board texture donk: connected board with two pair / straight
        if board_texture.connectivity >= 7:
            if made >= MadeHandType.BOTTOM_TWO_PAIR:
                bet_amount = min(hero_stack, pot * 0.45)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Donk bet: connected board, {made.name} for value+protection",
                    confidence=0.60,
                )

        return None

    def _evaluate_probe_bet(
        self,
        hand_strength: HandStrength,
        board_texture,
        pot: float,
        hero_stack: float,
        villain_profile: Optional[VillainProfile],
        spr: float,
    ) -> Optional[PostflopDecision]:
        """Evaluate turn probe bet (after villain flop check-back)."""
        made = hand_strength.made_hand
        draw = hand_strength.draw
        equity = hand_strength.equity_bucket

        # Float frequency: if villain floats a lot, our probe will get raised
        float_freq = 0.25
        if villain_profile is not None:
            float_freq = getattr(villain_profile.stats, "float_flop", 0.25)
        if float_freq > 0.45:
            return None

        # a. Value probe: middle pair+ has showdown value vs capped range
        if made >= MadeHandType.MIDDLE_PAIR and hand_strength.has_showdown_value:
            bet_amount = min(hero_stack, pot * 0.45)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Probe bet: value with {made.name}, villain capped (flop check-back)",
                confidence=0.63,
            )

        # b. Bluff probe: draws on capped range
        if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.FLUSH_DRAW_LOW, DrawType.OESD, DrawType.COMBO_DRAW, DrawType.COMBO_DRAW_NUT):
            fold_turn = 0.50
            if villain_profile is not None:
                fold_turn = getattr(villain_profile.stats, "fold_to_turn_cbet", 0.50)
            if fold_turn > 0.50:
                bet_amount = min(hero_stack, pot * 0.35)
                return PostflopDecision(
                    action="bet",
                    amount=bet_amount,
                    reasoning=f"Probe bluff: {draw.name}, villain capped + fold_to_turn={fold_turn:.0%}",
                    confidence=0.55,
                )

        # c. Merge probe: weak top pair / good middle pair (no free card)
        if made in (MadeHandType.TOP_PAIR_WEAK_KICKER, MadeHandType.MIDDLE_PAIR):
            bet_amount = min(hero_stack, pot * 0.40)
            return PostflopDecision(
                action="bet",
                amount=bet_amount,
                reasoning=f"Merge probe: {made.name}, deny free card after villain flop check",
                confidence=0.57,
            )

        return None

    def _villain_checked_back_flop(self, action_history: List[Dict]) -> bool:
        """Detect whether the villain checked back on the flop (no cbet)."""
        flop_actions = [
            a for a in action_history
            if a.get("street") == "flop" and a.get("actor", 0) != 0
        ]
        if not flop_actions:
            return False
        for action in flop_actions:
            if action.get("action") in ("bet", "raise"):
                return False
        return True

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
