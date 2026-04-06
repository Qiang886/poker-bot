"""Top-level PokerBot orchestrator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.card import Card
from src.position import Position
from src.hand_analysis import classify_hand
from src.board_analysis import analyze_board, analyze_range_advantage
from src.barrel_plan import BarrelPlan, create_barrel_plan
from src.preflop import PreflopEngine
from src.postflop import PostflopEngine
from src.opponent import VillainProfile
from src.weighted_range import build_range_combos, ComboWeight
from src.ranges import get_rfi_range
from src.icm import calculate_icm, icm_pressure, adjust_strategy_for_icm
from src.range_updater import RangeUpdater
from src.outs_calculator import OutsCalculator


@dataclass
class GameState:
    hero_hand: Tuple[Card, Card]
    board: List[Card]
    pot: float
    to_call: float
    hero_stack: float
    villain_stacks: List[float]
    hero_position: Position
    villain_positions: List[Position]
    street: str
    action_history: List[Dict]
    is_tournament: bool = False
    payouts: Optional[List[float]] = None
    num_players: int = 6


class PokerBot:
    def __init__(self, hero_position: Position, profile_name: str = "default") -> None:
        self.hero_position = hero_position
        self.profile_name = profile_name
        self.villain_profiles: Dict[int, VillainProfile] = {}
        self.preflop_engine = PreflopEngine()
        self.postflop_engine = PostflopEngine()
        self.current_barrel_plan: Optional[BarrelPlan] = None
        self._range_updater = RangeUpdater()
        self._outs_calc = OutsCalculator()
        # Per-hand villain range state (villain_id → range)
        self._villain_ranges: Dict[int, List[ComboWeight]] = {}

    def decide(self, game_state: GameState) -> Dict:
        gs = game_state
        result: Dict = {}

        if gs.street == "preflop":
            decision = self.preflop_engine.decide(
                hero_hand=gs.hero_hand,
                position=gs.hero_position,
                action_history=gs.action_history,
                pot=gs.pot,
                to_call=gs.to_call,
                hero_stack=gs.hero_stack,
                num_players=gs.num_players,
                villain_profiles=self.villain_profiles,
            )
            action, amount = decision.action, decision.amount
            # ICM adjustment for tournaments
            if gs.is_tournament and gs.payouts:
                all_stacks = [gs.hero_stack] + gs.villain_stacks
                hero_idx = 0
                pressure = icm_pressure(gs.hero_stack, all_stacks, gs.payouts, hero_idx)
                stb = gs.hero_stack / 1.0  # assume BB=1
                action, amount = adjust_strategy_for_icm(action, amount, gs.pot, pressure, stb)
            result = {"action": action, "amount": amount, "reasoning": decision.reasoning}

        else:
            # Build base villain ranges (RFI range → starting point)
            dead = list(gs.hero_hand) + list(gs.board)
            villain_range_combos: List[ComboWeight] = []
            for i, vpos in enumerate(gs.villain_positions):
                if i not in self._villain_ranges:
                    # Initialize from RFI range
                    vrange = get_rfi_range(vpos)
                    self._villain_ranges[i] = build_range_combos(vrange, dead)
                else:
                    # Remove dead cards from stored range
                    dead_set = set(dead)
                    self._villain_ranges[i] = [
                        cw for cw in self._villain_ranges[i]
                        if cw.combo[0] not in dead_set and cw.combo[1] not in dead_set
                    ]
                villain_range_combos += self._villain_ranges[i]

            hero_range_combos = build_range_combos(get_rfi_range(gs.hero_position), dead)

            # Apply range updates from action history for villain 0
            board_texture = analyze_board(gs.board)
            updated_villain_range = self._apply_range_updates(
                gs, board_texture, villain_range_combos
            )

            # Compute outs analysis for draw hands on flop/turn
            outs_analysis = None
            if gs.street in ("flop", "turn") and gs.board:
                hand_strength_check = classify_hand(gs.hero_hand, gs.board)
                from src.hand_analysis import DrawType
                if hand_strength_check.draw != DrawType.NONE:
                    outs_analysis = self._outs_calc.calculate_outs(
                        gs.hero_hand, gs.board, updated_villain_range,
                        simulations_per_out=150,
                    )

            # Create or reuse barrel plan on flop
            if gs.street == "flop" and self.current_barrel_plan is None:
                hand_strength = classify_hand(gs.hero_hand, gs.board)
                range_adv = analyze_range_advantage(hero_range_combos, updated_villain_range, gs.board)
                pos_label = "ip" if gs.hero_position in (Position.BTN, Position.CO) else "oop"
                spr = gs.hero_stack / gs.pot if gs.pot > 0 else 10.0
                vp0 = self.get_villain_profile(0)
                self.current_barrel_plan = create_barrel_plan(
                    hand_strength, board_texture, range_adv, pos_label, spr, vp0,
                    outs_analysis=outs_analysis,
                )

            is_multiway = len(gs.villain_positions) > 1
            villain_stack = gs.villain_stacks[0] if gs.villain_stacks else gs.hero_stack

            # Compute sizing_ratio from to_call / pot
            sizing_ratio = (gs.to_call / gs.pot) if gs.pot > 0 and gs.to_call > 0 else 0.0

            decision = self.postflop_engine.decide(
                hero_hand=gs.hero_hand,
                board=gs.board,
                pot=gs.pot,
                to_call=gs.to_call,
                hero_stack=gs.hero_stack,
                villain_stack=villain_stack,
                position=gs.hero_position,
                street=gs.street,
                villain_profile=self.get_villain_profile(0),
                hero_range=hero_range_combos,
                villain_range=updated_villain_range,
                barrel_plan=self.current_barrel_plan,
                is_multiway=is_multiway,
                sizing_ratio=sizing_ratio,
            )

            hand_strength = classify_hand(gs.hero_hand, gs.board)
            result = {
                "action": decision.action,
                "amount": decision.amount,
                "reasoning": decision.reasoning,
                "hand_strength": hand_strength,
                "barrel_plan": self.current_barrel_plan,
                "confidence": decision.confidence,
            }

        return result

    def _apply_range_updates(
        self,
        gs: "GameState",
        board_texture,
        villain_range: List[ComboWeight],
    ) -> List[ComboWeight]:
        """Apply RangeUpdater for each villain action in action_history.

        Action events should be dicts with these keys:
          - 'actor': int — 0 = hero (skip), 1+ = villain (update range)
          - 'action': str — "bet", "raise", "call", "check", "all_in", "fold"
          - 'amount': float — bet size
          - 'pot': float — pot at time of action (optional, falls back to gs.pot)
          - 'street': str — "flop", "turn", "river" (optional, falls back to gs.street)
          - 'is_ip': bool — whether villain is in position (optional, default True)
        """
        updated = list(villain_range)
        if not gs.action_history or not gs.board:
            return updated

        for event in gs.action_history:
            if not isinstance(event, dict):
                continue

            # Determine actor: 0 = hero (skip), anything else = villain
            actor = event.get("actor", event.get("villain_id"))
            if actor == 0:
                continue  # hero action – skip

            action = event.get("action", "")
            if not action:
                continue

            amount = float(event.get("amount", 0.0))
            pot = float(event.get("pot", gs.pot))
            sizing_ratio = (amount / pot) if pot > 0 else 0.0
            street = event.get("street", gs.street)
            is_ip = event.get("is_ip", True)

            action_lc = action.lower().replace("-", "_")
            if action_lc in ("bet", "raise", "all_in", "call", "check", "fold"):
                updated = self._range_updater.update_range_after_action(
                    updated,
                    list(gs.board),
                    action_lc,
                    sizing_ratio,
                    street,
                    board_texture,
                    is_ip,
                )

        return updated

    def update_villain_action(
        self,
        villain_id: int,
        action: str,
        amount: float,
        position: Position,
        street: str,
        pot: float,
        board: Optional[List[Card]] = None,
        board_texture=None,
        is_ip: bool = True,
    ) -> None:
        profile = self.get_villain_profile(villain_id)
        profile.update_action(street, action, position, amount, pot)

        # Also update villain's stored range if we have board info
        if board is not None and villain_id in self._villain_ranges:
            if board_texture is None:
                board_texture = analyze_board(board)
            sizing_ratio = (amount / pot) if pot > 0 else 0.0
            action_lc = action.lower().replace("-", "_")
            self._villain_ranges[villain_id] = self._range_updater.update_range_after_action(
                self._villain_ranges[villain_id],
                board,
                action_lc,
                sizing_ratio,
                street,
                board_texture,
                is_ip,
            )

    def reset_hand(self) -> None:
        self.current_barrel_plan = None
        self._villain_ranges = {}

    def get_villain_profile(self, villain_id: int) -> VillainProfile:
        if villain_id not in self.villain_profiles:
            self.villain_profiles[villain_id] = VillainProfile()
        return self.villain_profiles[villain_id]
