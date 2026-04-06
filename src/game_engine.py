"""游戏引擎：发牌、下注规则、底池管理（Heads-Up 6-max NLHE）。"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.card import Card, Rank, Suit

# ---------------------------------------------------------------------------
# 花色 Unicode 符号
# ---------------------------------------------------------------------------
SUIT_UNICODE = {
    Suit.SPADES:   "♠",
    Suit.HEARTS:   "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS:    "♣",
}

RANK_DISPLAY = {
    Rank.TWO:   "2",
    Rank.THREE: "3",
    Rank.FOUR:  "4",
    Rank.FIVE:  "5",
    Rank.SIX:   "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE:  "9",
    Rank.TEN:   "T",
    Rank.JACK:  "J",
    Rank.QUEEN: "Q",
    Rank.KING:  "K",
    Rank.ACE:   "A",
}

STREETS = ["preflop", "flop", "turn", "river"]


def card_display(card: Card) -> str:
    """返回 [A♠] 格式的牌面字符串。"""
    return f"[{RANK_DISPLAY[card.rank]}{SUIT_UNICODE[card.suit]}]"


def cards_display(cards: List[Card]) -> str:
    """返回多张牌的显示字符串，以空格分隔。"""
    return " ".join(card_display(c) for c in cards)


# ---------------------------------------------------------------------------
# 牌组
# ---------------------------------------------------------------------------

class GameDeck:
    """标准 52 张牌组，每手牌重新洗牌。"""

    def __init__(self) -> None:
        self._cards: List[Card] = [Card(r, s) for r in Rank for s in Suit]
        random.shuffle(self._cards)

    def deal(self) -> Card:
        return self._cards.pop()

    def deal_n(self, n: int) -> List[Card]:
        return [self.deal() for _ in range(n)]

    def __len__(self) -> int:
        return len(self._cards)


# ---------------------------------------------------------------------------
# 游戏状态数据类
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    name: str           # "你" or "Bot"
    stack: float        # 当前筹码（BB 为单位）
    hole_cards: List[Card] = field(default_factory=list)
    bet_street: float = 0.0   # 本街已投入
    total_invested: float = 0.0  # 本手牌已总投入（含盲注）
    folded: bool = False
    is_allin: bool = False


@dataclass
class HandResult:
    winner: str           # "你" / "Bot" / "平局"
    amount: float         # 赢得筹码数量（BB）
    showdown: bool        # 是否到摊牌
    player_best: Optional[str] = None   # 你的最佳牌型描述
    bot_best: Optional[str] = None      # bot 的最佳牌型描述
    reason: str = ""      # 弃牌理由或摊牌说明


@dataclass
class ActionRecord:
    """记录一次行动，供 GameState action_history 使用。"""
    player: str       # "hero" / "villain"
    action: str       # "fold"/"call"/"check"/"raise"/"all-in"
    amount: float     # raise 到多少（总量，不是增量）
    street: str


# ---------------------------------------------------------------------------
# 下注规则辅助
# ---------------------------------------------------------------------------

def clamp_raise(amount: float, min_raise: float, max_raise: float) -> float:
    """将 raise 金额夹在 [min_raise, max_raise] 之间。"""
    return max(min_raise, min(amount, max_raise))


def calc_min_raise(last_raise_total: float, last_raise_increment: float, to_call: float) -> float:
    """
    最小加注额 = 上一次加注的增量（或至少 1BB）。
    返回的是 raise 到的总金额（相对于当前玩家已投入的）。
    """
    min_increment = max(last_raise_increment, 1.0)
    return to_call + min_increment


# ---------------------------------------------------------------------------
# 手牌牌型描述（用于摊牌展示）
# ---------------------------------------------------------------------------

def hand_rank_description(eval_result: Tuple) -> str:
    """将 evaluate_7() 的结果转成中文描述。"""
    from src.evaluator import HandRank
    hand_rank, tiebreakers = eval_result
    rank_names = {
        HandRank.HIGH_CARD:       "高牌",
        HandRank.ONE_PAIR:        "一对",
        HandRank.TWO_PAIR:        "两对",
        HandRank.THREE_OF_A_KIND: "三条",
        HandRank.STRAIGHT:        "顺子",
        HandRank.FLUSH:           "同花",
        HandRank.FULL_HOUSE:      "葫芦",
        HandRank.FOUR_OF_A_KIND:  "四条",
        HandRank.STRAIGHT_FLUSH:  "同花顺",
    }
    return rank_names.get(hand_rank, str(hand_rank))
