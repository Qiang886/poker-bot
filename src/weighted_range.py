"""Weighted range utilities: expand notations, build combo lists, blocker scoring."""

from dataclasses import dataclass, field
from typing import List, Tuple, Set

from src.card import Card, Rank, Suit, RANK_FROM_CHAR, RANK_CHAR


@dataclass
class ComboWeight:
    combo: Tuple[Card, Card]
    weight: float = 1.0


class HandNotation:
    """Parse hand notations like 'AKs', 'AKo', 'AA'."""

    def __init__(self, notation: str) -> None:
        self.notation = notation.strip()
        if len(self.notation) == 2:
            if self.notation[0] not in RANK_FROM_CHAR or self.notation[1] not in RANK_FROM_CHAR:
                raise ValueError(f"Invalid pair notation: {notation}")
            if self.notation[0] != self.notation[1]:
                raise ValueError(f"Two-char notation must be a pair: {notation}")
            self.is_pair = True
            self.rank1 = RANK_FROM_CHAR[self.notation[0]]
            self.rank2 = self.rank1
            self.suited = False
            self.offsuit = False
        elif len(self.notation) == 3:
            self.is_pair = False
            self.rank1 = RANK_FROM_CHAR[self.notation[0]]
            self.rank2 = RANK_FROM_CHAR[self.notation[1]]
            qualifier = self.notation[2].lower()
            if qualifier not in ('s', 'o'):
                raise ValueError(f"Qualifier must be 's' or 'o': {notation}")
            self.suited = qualifier == 's'
            self.offsuit = qualifier == 'o'
        else:
            raise ValueError(f"Invalid notation length: {notation}")


def expand_hand_notation(notation: str) -> List[Tuple[Card, Card]]:
    """Expand a hand notation string into all specific card combos.

    'AA'  -> 6 combos (all pairs)
    'AKs' -> 4 suited combos
    'AKo' -> 12 offsuit combos
    """
    hn = HandNotation(notation)
    result: List[Tuple[Card, Card]] = []
    all_suits = list(Suit)

    if hn.is_pair:
        for i, s1 in enumerate(all_suits):
            for s2 in all_suits[i + 1:]:
                result.append((Card(hn.rank1, s1), Card(hn.rank1, s2)))
    elif hn.suited:
        for s in all_suits:
            result.append((Card(hn.rank1, s), Card(hn.rank2, s)))
    else:  # offsuit
        for s1 in all_suits:
            for s2 in all_suits:
                if s1 != s2:
                    result.append((Card(hn.rank1, s1), Card(hn.rank2, s2)))

    return result


def build_range_combos(hands: Set[str], dead_cards: List[Card]) -> List[ComboWeight]:
    """Build weighted combo list from hand notation set, removing dead combos."""
    dead_set: Set[Card] = set(dead_cards)
    result: List[ComboWeight] = []
    for hand in hands:
        try:
            combos = expand_hand_notation(hand)
        except ValueError:
            continue
        for combo in combos:
            if combo[0] not in dead_set and combo[1] not in dead_set:
                result.append(ComboWeight(combo=combo, weight=1.0))
    return result


def filter_range_by_action(combos: List[ComboWeight], action_line: str) -> List[ComboWeight]:
    """Adjust combo weights based on observed action line.

    action_line: 'open', '3bet', '4bet', 'call', 'limp'
    """
    result: List[ComboWeight] = []
    action = action_line.lower()

    for cw in combos:
        c1, c2 = cw.combo
        r1, r2 = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
        is_pair = r1 == r2
        is_suited = c1.suit == c2.suit
        new_weight = cw.weight

        if action == "limp":
            # Remove premium hands from limp range
            if is_pair and r1 >= Rank.JACK:
                new_weight = 0.0
            elif r1 == Rank.ACE and r2 >= Rank.KING:
                new_weight = 0.0
            else:
                new_weight = cw.weight * 0.8

        elif action == "4bet":
            # 4bet: only premiums and bluff aces
            if is_pair and r1 >= Rank.QUEEN:
                new_weight = cw.weight
            elif r1 == Rank.ACE and r2 == Rank.KING:
                new_weight = cw.weight
            elif r1 == Rank.ACE and r2 <= Rank.FIVE and is_suited:
                new_weight = cw.weight * 0.7  # bluff
            else:
                new_weight = 0.0

        elif action == "3bet":
            # 3bet: value + polarised bluffs
            if is_pair and r1 >= Rank.JACK:
                new_weight = cw.weight
            elif r1 == Rank.ACE and (r2 >= Rank.QUEEN or (r2 <= Rank.FIVE and is_suited)):
                new_weight = cw.weight
            elif r1 == Rank.KING and r2 == Rank.KING:
                new_weight = cw.weight
            else:
                new_weight = 0.0

        elif action == "call":
            # Calling range: remove pure air
            if not is_pair and r1 <= Rank.SEVEN and r2 <= Rank.FIVE:
                new_weight = 0.0

        # "open" -> no adjustment needed

        if new_weight > 0.0:
            result.append(ComboWeight(combo=cw.combo, weight=new_weight))

    return result


def calculate_blocker_score(
    hero_hand: Tuple[Card, Card],
    villain_range: List[ComboWeight],
    board: List[Card],
) -> float:
    """Return 0-1 score; higher means hero blocks more of villain's strong combos."""
    if not villain_range:
        return 0.0

    hero_cards = set(hero_hand)
    total_weight = sum(cw.weight for cw in villain_range)
    if total_weight == 0:
        return 0.0

    blocked_weight = sum(
        cw.weight
        for cw in villain_range
        if cw.combo[0] in hero_cards or cw.combo[1] in hero_cards
    )

    # Normalise: a perfect blocker would block ~half the range (average)
    raw = blocked_weight / total_weight
    # Scale to 0-1
    return min(1.0, raw * 4.0)


def count_combos(hands: Set[str]) -> int:
    """Count the total number of specific combos in a set of hand notations."""
    total = 0
    for hand in hands:
        try:
            total += len(expand_hand_notation(hand))
        except ValueError:
            pass
    return total
