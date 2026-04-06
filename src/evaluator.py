"""Hand evaluator for 5-card and 7-card poker hands."""

from collections import Counter
from enum import IntEnum
from itertools import combinations
from typing import List, Tuple

from src.card import Card, Rank


class HandRank(IntEnum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


EvalResult = Tuple[HandRank, tuple]


def evaluate_5(cards: List[Card]) -> EvalResult:
    """Evaluate a 5-card hand. Returns (HandRank, tiebreaker_tuple)."""
    ranks = sorted([int(c.rank) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif set(unique_ranks) == {14, 2, 3, 4, 5}:
            # Ace-low straight: A-2-3-4-5
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return (HandRank.STRAIGHT_FLUSH, (straight_high,))

    if counts[0] == 4:
        quad_rank = max(r for r, c in rank_counts.items() if c == 4)
        kicker = max(r for r, c in rank_counts.items() if c == 1)
        return (HandRank.FOUR_OF_A_KIND, (quad_rank, kicker))

    if counts[0] == 3 and counts[1] == 2:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        return (HandRank.FULL_HOUSE, (trip_rank, pair_rank))

    if is_flush:
        return (HandRank.FLUSH, tuple(ranks))

    if is_straight:
        return (HandRank.STRAIGHT, (straight_high,))

    if counts[0] == 3:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandRank.THREE_OF_A_KIND, (trip_rank, *kickers))

    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = max(r for r, c in rank_counts.items() if c == 1)
        return (HandRank.TWO_PAIR, (pairs[0], pairs[1], kicker))

    if counts[0] == 2:
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandRank.ONE_PAIR, (pair_rank, *kickers))

    return (HandRank.HIGH_CARD, tuple(ranks))


def evaluate_7(cards: List[Card]) -> EvalResult:
    """Evaluate a 7-card hand by checking all C(7,5) combinations."""
    best: EvalResult = (HandRank.HIGH_CARD, (0,))
    for five in combinations(cards, 5):
        result = evaluate_5(list(five))
        if result > best:
            best = result
    return best
