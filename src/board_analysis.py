"""Board texture analysis and range-advantage determination."""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import List

from src.card import Card, Rank, Suit
from src.evaluator import evaluate_7, HandRank
from src.weighted_range import ComboWeight


@dataclass
class BoardTexture:
    wetness: int          # 0-10
    pairedness: int       # 0=unpaired, 1=paired, 2=trips on board
    monotone: bool
    two_tone: bool
    rainbow: bool
    highest_card: Rank
    connectivity: int     # 0-10
    flush_possible: bool
    straight_possible: bool
    paired: bool


class NutAdvantage(Enum):
    HERO = "hero"
    VILLAIN = "villain"
    NEUTRAL = "neutral"


@dataclass
class RangeAdvantage:
    nut_advantage: NutAdvantage
    equity_advantage: float             # -1 to 1  (positive = hero advantage)
    recommended_bet_frequency: float    # 0-1


def analyze_board(board: List[Card]) -> BoardTexture:
    """Derive texture metrics from the current board cards."""
    if not board:
        return BoardTexture(
            wetness=0, pairedness=0, monotone=False, two_tone=False,
            rainbow=True, highest_card=Rank.TWO, connectivity=0,
            flush_possible=False, straight_possible=False, paired=False,
        )

    ranks = [c.rank for c in board]
    suits = [c.suit for c in board]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    highest_card = Rank(max(ranks))
    max_same_rank = max(rank_counts.values())
    pairedness = max_same_rank - 1   # 0/1/2
    paired = max_same_rank >= 2

    max_same_suit = max(suit_counts.values())
    n = len(board)
    monotone = (max_same_suit == n and n >= 3)
    two_tone = (max_same_suit == 2 and n >= 3)
    rainbow = (max_same_suit == 1)
    flush_possible = (max_same_suit >= 2)

    unique_ranks = sorted(set(ranks))
    gaps = 0
    if len(unique_ranks) > 1:
        for i in range(len(unique_ranks) - 1):
            gaps += unique_ranks[i + 1] - unique_ranks[i] - 1
    connectivity = max(0, 10 - gaps * 2) if n >= 2 else 0

    straight_possible = (
        len(unique_ranks) >= 3
        and max(unique_ranks) - min(unique_ranks) <= 4
    )

    wetness = 0
    if max_same_suit >= 2:
        wetness += 3
    if monotone:
        wetness += 2
    wetness += min(connectivity // 2, 3)
    if highest_card <= Rank.NINE:
        wetness += 2   # low connected boards are wetter
    wetness = min(10, wetness)

    return BoardTexture(
        wetness=wetness, pairedness=pairedness, monotone=monotone,
        two_tone=two_tone, rainbow=rainbow, highest_card=highest_card,
        connectivity=connectivity, flush_possible=flush_possible,
        straight_possible=straight_possible, paired=paired,
    )


def _range_strength(combo_list: List[ComboWeight], board: List[Card]) -> float:
    """Average hand-rank score for a range on a board (fast heuristic)."""
    if not combo_list or len(board) < 3:
        return 0.5
    total_weight = 0.0
    score_sum = 0.0
    for cw in combo_list:
        all_cards = list(cw.combo) + list(board)
        rank_val, _ = evaluate_7(all_cards)
        score_sum += cw.weight * (int(rank_val) / 9.0)
        total_weight += cw.weight
    if total_weight == 0:
        return 0.5
    return score_sum / total_weight


def _count_nuts(combo_list: List[ComboWeight], board: List[Card]) -> float:
    """Fraction of strong hands (flush or better) in range."""
    if not combo_list or len(board) < 3:
        return 0.0
    total_weight = 0.0
    nut_weight = 0.0
    for cw in combo_list:
        all_cards = list(cw.combo) + list(board)
        rank_val, _ = evaluate_7(all_cards)
        if rank_val >= HandRank.FLUSH:
            nut_weight += cw.weight
        total_weight += cw.weight
    if total_weight == 0:
        return 0.0
    return nut_weight / total_weight


def analyze_range_advantage(
    hero_range: List[ComboWeight],
    villain_range: List[ComboWeight],
    board: List[Card],
) -> RangeAdvantage:
    """Determine who holds the range/nut advantage and suggest cbet frequency."""
    if not hero_range and not villain_range:
        return RangeAdvantage(NutAdvantage.NEUTRAL, 0.0, 0.5)

    hero_strength = _range_strength(hero_range, board)
    villain_strength = _range_strength(villain_range, board)
    equity_adv = hero_strength - villain_strength   # -1 to 1

    hero_nuts = _count_nuts(hero_range, board)
    villain_nuts = _count_nuts(villain_range, board)

    if hero_nuts > villain_nuts + 0.05:
        nut_adv = NutAdvantage.HERO
    elif villain_nuts > hero_nuts + 0.05:
        nut_adv = NutAdvantage.VILLAIN
    else:
        nut_adv = NutAdvantage.NEUTRAL

    # Bet frequency heuristic
    if nut_adv == NutAdvantage.HERO:
        freq = min(0.85, 0.60 + equity_adv * 0.5)
    elif nut_adv == NutAdvantage.VILLAIN:
        freq = max(0.20, 0.40 + equity_adv * 0.3)
    else:
        freq = max(0.30, min(0.70, 0.50 + equity_adv * 0.4))

    return RangeAdvantage(
        nut_advantage=nut_adv,
        equity_advantage=equity_adv,
        recommended_bet_frequency=freq,
    )


def get_board_texture_description(texture: BoardTexture) -> str:
    """Human-readable description of a board texture."""
    parts = []

    if texture.wetness >= 8:
        parts.append("very wet")
    elif texture.wetness >= 5:
        parts.append("wet")
    elif texture.wetness >= 3:
        parts.append("semi-dry")
    else:
        parts.append("dry")

    if texture.monotone:
        parts.append("monotone")
    elif texture.two_tone:
        parts.append("two-tone")
    else:
        parts.append("rainbow")

    if texture.paired:
        parts.append("paired")

    if texture.straight_possible:
        parts.append("straight draws possible")
    if texture.flush_possible and not texture.monotone:
        parts.append("flush draws possible")

    return ", ".join(parts)
