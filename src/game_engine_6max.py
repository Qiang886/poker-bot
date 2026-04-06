"""6-max 游戏引擎：支持 6 个玩家、Side Pot 计算、多人下注轮。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.card import Card, Rank, Suit
from src.position import Position
from src.game_engine import (
    GameDeck,
    card_display,
    cards_display,
    hand_rank_description,
    clamp_raise,
    calc_min_raise,
    STREETS,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SB_AMOUNT = 0.5
BB_AMOUNT = 1.0

# 6-max 位置列表（按桌子顺序）
SIX_MAX_POSITIONS: List[Position] = [
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
    Position.SB,
    Position.BB,
]

POSITION_NAMES = {
    Position.UTG: "UTG",
    Position.HJ:  "HJ",
    Position.CO:  "CO",
    Position.BTN: "BTN",
    Position.SB:  "SB",
    Position.BB:  "BB",
}

# 翻前行动顺序：UTG → HJ → CO → BTN → SB → BB
PREFLOP_ORDER: List[Position] = [
    Position.UTG, Position.HJ, Position.CO,
    Position.BTN, Position.SB, Position.BB,
]

# 翻后行动顺序：SB → BB → UTG → HJ → CO → BTN
POSTFLOP_ORDER: List[Position] = [
    Position.SB, Position.BB, Position.UTG,
    Position.HJ, Position.CO, Position.BTN,
]


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class PlayerState6:
    name: str
    position: Position
    stack: float
    hole_cards: List[Card] = field(default_factory=list)
    bet_street: float = 0.0        # 本街已投入
    total_invested: float = 0.0    # 本手总投入（含盲注）
    folded: bool = False
    is_allin: bool = False
    is_human: bool = False

    @property
    def pos_name(self) -> str:
        return POSITION_NAMES[self.position]

    @property
    def active(self) -> bool:
        """还在底池中（未弃牌）。"""
        return not self.folded


@dataclass
class SidePot:
    amount: float
    eligible_players: List[int]   # 玩家在 players 列表中的索引


@dataclass
class HandResult6:
    pot_winners: List[Tuple[int, float, str]]  # (player_idx, amount, description)
    showdown: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Side Pot 计算
# ---------------------------------------------------------------------------

def calculate_side_pots(players: List[PlayerState6]) -> List[SidePot]:
    """
    根据每个玩家的总投入计算 main pot 和 side pots。
    返回 SidePot 列表，每个包含 amount 和 eligible_players（索引）。
    """
    side_pots: List[SidePot] = []
    prev_level = 0.0

    # 把弃牌玩家的投入也纳入计算（但他们不参与分配）
    all_invested = [(i, p.total_invested) for i, p in enumerate(players)]
    all_invested.sort(key=lambda x: x[1])

    # 找到所有唯一的 all-in 级别
    levels = sorted(set(ti for _, ti in all_invested if ti > 0))

    for level in levels:
        eligible = [i for i, p in enumerate(players) if not p.folded and p.total_invested >= level]
        if not eligible:
            continue
        # 每个玩家在这一层的贡献
        pot_amount = 0.0
        for i, p in enumerate(players):
            contribution = min(p.total_invested, level) - min(p.total_invested, prev_level)
            pot_amount += max(0.0, contribution)

        if pot_amount > 0.001:
            side_pots.append(SidePot(amount=pot_amount, eligible_players=eligible))
        prev_level = level

    if not side_pots:
        # 简单情况：没有 all-in，全部给活跃玩家
        eligible = [i for i, p in enumerate(players) if not p.folded]
        total = sum(p.total_invested for p in players)
        if total > 0 and eligible:
            side_pots.append(SidePot(amount=total, eligible_players=eligible))

    return side_pots


# ---------------------------------------------------------------------------
# 下注轮
# ---------------------------------------------------------------------------

def betting_round_6max(
    street: str,
    players: List[PlayerState6],
    action_order: List[int],   # 按位置顺序的玩家索引列表（含已弃牌/all-in 的）
    pot: float,
    action_history: list,
    last_raise_increment: float,
    on_player_action,          # callable(player_idx, to_call, pot, min_raise, last_raise_increment) -> (action, amount, new_pot, new_last_raise_increment)
) -> Tuple[float, float, bool]:
    """
    6-max 下注轮。
    标准扑克加注规则：加注后从加注者之后的玩家开始重新行动，
    直到回到加注者（所有人行动完且下注额相等）为止。

    返回 (pot, last_raise_increment, game_continues)
    game_continues=False 表示只剩1人活跃（其余全弃牌）
    """
    # 重置本街下注（翻后）
    if street != "preflop":
        for p in players:
            p.bet_street = 0.0
        last_raise_increment = BB_AMOUNT

    n = len(action_order)

    def max_bet() -> float:
        return max((p.bet_street for p in players if not p.folded), default=0.0)

    def active_non_allin() -> List[int]:
        return [idx for idx in action_order if not players[idx].folded and not players[idx].is_allin]

    def all_equal_and_acted(acted: set) -> bool:
        mb = max_bet()
        eligible = active_non_allin()
        return all(
            idx in acted and abs(players[idx].bet_street - mb) < 0.001
            for idx in eligible
        )

    acted: set = set()    # 本轮已行动的玩家
    i = 0                 # 当前在 action_order 中的扫描指针

    while True:
        # 终止检查
        eligible = active_non_allin()
        if not eligible:
            break
        if all_equal_and_acted(acted):
            break

        idx = action_order[i % n]
        p = players[idx]

        # 跳过弃牌或 all-in 的
        if p.folded or p.is_allin:
            i += 1
            continue

        mb = max_bet()
        to_call = max(0.0, mb - p.bet_street)

        # 该玩家是否需要行动？
        # 需要行动：未行动过，或下注额小于 max_bet
        if idx in acted and abs(p.bet_street - mb) < 0.001:
            i += 1
            continue

        min_raise_val = calc_min_raise(mb, last_raise_increment, to_call)

        # 执行行动
        action, amount, pot, last_raise_increment = on_player_action(
            idx, to_call, pot, min_raise_val, last_raise_increment
        )

        acted.add(idx)

        if action == "fold":
            p.folded = True
            active_players = [pp for pp in players if not pp.folded]
            if len(active_players) <= 1:
                # 把本街下注额累加到总投入后返回
                for pl in players:
                    pl.total_invested += pl.bet_street
                return pot, last_raise_increment, False
            i += 1
            continue

        if action in ("raise", "allin"):
            # 加注后：从加注者之后开始，所有其他活跃非 all-in 玩家重新行动
            # 只保留加注者在 acted 中
            acted = {idx}
            # 从加注者之后开始扫描
            raiser_pos = action_order.index(idx)
            i = (raiser_pos + 1) % n
            continue

        i += 1

    # 把本街下注额累加到总投入
    for p in players:
        p.total_invested += p.bet_street

    return pot, last_raise_increment, True
