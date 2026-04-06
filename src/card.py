"""Card primitives: Suit, Rank, Card, Deck, helpers."""

from enum import IntEnum
import random
from typing import List


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


RANK_FROM_CHAR = {
    '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
    '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
    'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE,
}
SUIT_FROM_CHAR = {
    'c': Suit.CLUBS, 'd': Suit.DIAMONDS, 'h': Suit.HEARTS, 's': Suit.SPADES,
}
RANK_CHAR = {v: k for k, v in RANK_FROM_CHAR.items()}
SUIT_CHAR = {v: k for k, v in SUIT_FROM_CHAR.items()}


class Card:
    __slots__ = ('rank', 'suit')

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self.rank = Rank(rank)
        self.suit = Suit(suit)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return int(self.rank) * 4 + int(self.suit)

    def __repr__(self) -> str:
        return f"{RANK_CHAR[self.rank]}{SUIT_CHAR[self.suit]}"

    def __lt__(self, other: 'Card') -> bool:
        return self.rank < other.rank


def card_from_str(s: str) -> Card:
    """Parse a 2-character string like 'As' or 'Th' into a Card."""
    s = s.strip()
    return Card(RANK_FROM_CHAR[s[0].upper()], SUIT_FROM_CHAR[s[1].lower()])


def cards_from_str(s: str) -> List[Card]:
    """Parse a string like 'AsKh' into [Card(ACE,SPADES), Card(KING,HEARTS)]."""
    s = s.strip()
    return [card_from_str(s[i:i + 2]) for i in range(0, len(s), 2)]


class Deck:
    """Standard 52-card deck."""

    def __init__(self) -> None:
        self._cards: List[Card] = [Card(r, s) for r in Rank for s in Suit]
        random.shuffle(self._cards)

    def shuffle(self) -> None:
        random.shuffle(self._cards)

    def deal(self) -> Card:
        return self._cards.pop()

    def remove(self, cards: List[Card]) -> None:
        for c in cards:
            self._cards.remove(c)

    def __len__(self) -> int:
        return len(self._cards)


FULL_DECK: List[Card] = [Card(r, s) for r in Rank for s in Suit]
