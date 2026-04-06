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
            # Build ranges (simplified: use RFI range as villain default)
            dead = list(gs.hero_hand) + list(gs.board)
            villain_range_combos: List[ComboWeight] = []
            for i, vpos in enumerate(gs.villain_positions):
                vp = self.get_villain_profile(i)
                vrange = get_rfi_range(vpos)
                villain_range_combos += build_range_combos(vrange, dead)

            hero_range_combos = build_range_combos(get_rfi_range(gs.hero_position), dead)

            # Create or reuse barrel plan on flop
            if gs.street == "flop" and self.current_barrel_plan is None:
                hand_strength = classify_hand(gs.hero_hand, gs.board)
                board_texture = analyze_board(gs.board)
                range_adv = analyze_range_advantage(hero_range_combos, villain_range_combos, gs.board)
                pos_label = "ip" if gs.hero_position in (Position.BTN, Position.CO) else "oop"
                spr = gs.hero_stack / gs.pot if gs.pot > 0 else 10.0
                vp0 = self.get_villain_profile(0)
                self.current_barrel_plan = create_barrel_plan(
                    hand_strength, board_texture, range_adv, pos_label, spr, vp0,
                )

            is_multiway = len(gs.villain_positions) > 1
            villain_stack = gs.villain_stacks[0] if gs.villain_stacks else gs.hero_stack

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
                villain_range=villain_range_combos,
                barrel_plan=self.current_barrel_plan,
                is_multiway=is_multiway,
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

    def update_villain_action(
        self,
        villain_id: int,
        action: str,
        amount: float,
        position: Position,
        street: str,
        pot: float,
    ) -> None:
        profile = self.get_villain_profile(villain_id)
        profile.update_action(street, action, position, amount, pot)

    def reset_hand(self) -> None:
        self.current_barrel_plan = None

    def get_villain_profile(self, villain_id: int) -> VillainProfile:
        if villain_id not in self.villain_profiles:
            self.villain_profiles[villain_id] = VillainProfile()
        return self.villain_profiles[villain_id]
