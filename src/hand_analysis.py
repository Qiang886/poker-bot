"""Hand classification and draw detection for postflop analysis."""

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Dict, List, Set, Tuple

from src.card import Card, Rank, Suit, FULL_DECK
from src.evaluator import evaluate_5, evaluate_7, HandRank


class MadeHandType(IntEnum):
    NO_PAIR = 0
    ACE_HIGH = 1
    BOTTOM_PAIR = 2
    WEAK_MIDDLE_PAIR = 3
    MIDDLE_PAIR = 4
    TOP_PAIR_WEAK_KICKER = 5
    TOP_PAIR_GOOD_KICKER = 6
    TOP_PAIR_TOP_KICKER = 7
    OVERPAIR_SMALL = 8
    OVERPAIR_BIG = 9
    BOTTOM_TWO_PAIR = 10
    TOP_TWO_PAIR = 11
    TRIPS_SET = 12
    TRIPS_BOARD = 13
    STRAIGHT_NON_NUT = 14
    STRAIGHT_NUT = 15
    FLUSH_LOW = 16
    FLUSH_NUT = 17
    FULL_HOUSE = 18
    QUADS = 19
    STRAIGHT_FLUSH = 20


class DrawType(IntEnum):
    NONE = 0
    GUTSHOT = 1
    OESD = 2
    FLUSH_DRAW_LOW = 3
    FLUSH_DRAW_NUT = 4
    COMBO_DRAW = 5
    COMBO_DRAW_NUT = 6


@dataclass
class HandStrength:
    made_hand: MadeHandType
    draw: DrawType
    equity_bucket: float    # 0-1
    is_vulnerable: bool
    has_showdown_value: bool


# ---------------------------------------------------------------------------
# Draw detection helpers
# ---------------------------------------------------------------------------

def _flush_draw_type(hero_hand: Tuple[Card, Card], board: List[Card]) -> DrawType:
    """Detect flush draw and whether it is the nut flush draw."""
    all_cards = list(hero_hand) + list(board)
    suit_counts: Counter = Counter(c.suit for c in all_cards)
    for suit, count in suit_counts.items():
        if count == 4:
            # At least one hero card must contribute
            hero_suits = {hero_hand[0].suit, hero_hand[1].suit}
            if suit not in hero_suits:
                continue
            # Find all cards of this suit
            suit_cards = sorted(
                [c for c in all_cards if c.suit == suit],
                key=lambda c: c.rank, reverse=True,
            )
            # Check if hero holds the highest card of that suit
            hero_suit_cards = [c for c in suit_cards if c in set(hero_hand)]
            if hero_suit_cards and hero_suit_cards[0].rank == suit_cards[0].rank:
                # Check if ace-high flush draw
                if suit_cards[0].rank == Rank.ACE:
                    return DrawType.FLUSH_DRAW_NUT
                # Check if no higher card is possible that would beat this
                return DrawType.FLUSH_DRAW_NUT if suit_cards[0].rank >= Rank.KING else DrawType.FLUSH_DRAW_LOW
            return DrawType.FLUSH_DRAW_LOW
    return DrawType.NONE


_ACE_HIGH = 14  # numeric value of Rank.ACE
_ACE_LOW = 1    # low-ace alias used in A-2-3-4-5 straights


def _straight_draw_type(hero_hand: Tuple[Card, Card], board: List[Card]) -> DrawType:
    """Detect OESD or gutshot straight draw."""
    all_ranks = sorted({int(c.rank) for c in (list(hero_hand) + list(board))})
    # Also treat ace as low (1) for wheel draws
    if _ACE_HIGH in all_ranks:
        all_ranks_with_low_ace = sorted(set(all_ranks) | {_ACE_LOW})
    else:
        all_ranks_with_low_ace = all_ranks

    hero_ranks = {int(c.rank) for c in hero_hand}
    # Also add low ace for wheel draws
    if _ACE_HIGH in hero_ranks:
        hero_ranks = hero_ranks | {_ACE_LOW}

    oesd = False
    gutshot = False

    for high in range(5, 15):
        window = set(range(high - 4, high + 1))
        in_window = window & set(all_ranks_with_low_ace)
        if len(in_window) == 4:
            # We have 4 of the 5 cards needed
            missing = window - set(all_ranks_with_low_ace)
            if not missing:
                continue  # already a straight
            missing_rank = list(missing)[0]
            # Hero must contribute to the draw
            if not (window & hero_ranks):
                continue
            # OESD: missing card is at the top or bottom of window
            if missing_rank == high or missing_rank == high - 4:
                oesd = True
            else:
                gutshot = True

    if oesd:
        return DrawType.OESD
    if gutshot:
        return DrawType.GUTSHOT
    return DrawType.NONE


def detect_draws(hero_hand: Tuple[Card, Card], board: List[Card]) -> DrawType:
    """Detect the best draw type in hero's hand on the current board."""
    if len(board) > 4:
        return DrawType.NONE  # River: no draws

    flush = _flush_draw_type(hero_hand, board)
    straight = _straight_draw_type(hero_hand, board)

    has_flush = flush != DrawType.NONE
    has_straight = straight != DrawType.NONE

    if has_flush and has_straight:
        if flush == DrawType.FLUSH_DRAW_NUT:
            return DrawType.COMBO_DRAW_NUT
        return DrawType.COMBO_DRAW
    if has_flush:
        return flush
    if has_straight:
        return straight
    return DrawType.NONE


# ---------------------------------------------------------------------------
# Nut-hand detection
# ---------------------------------------------------------------------------

def is_nut_hand(hero_hand: Tuple[Card, Card], board: List[Card]) -> bool:
    """Return True if hero's made hand is the best possible hand on this board."""
    if len(board) < 3:
        return False

    hero_result = evaluate_7(list(hero_hand) + list(board))
    dead = set(hero_hand) | set(board)

    # Try all possible 2-card villain hands
    remaining = [c for c in FULL_DECK if c not in dead]
    for villain_combo in combinations(remaining, 2):
        villain_result = evaluate_7(list(villain_combo) + list(board))
        if villain_result > hero_result:
            return False
    return True


# ---------------------------------------------------------------------------
# Backdoor draw detection
# ---------------------------------------------------------------------------

def has_backdoor_draw(hero_hand: Tuple[Card, Card], board: List[Card]) -> bool:
    """Return True if hero has a backdoor flush or straight draw (on the flop)."""
    if len(board) != 3:
        return False

    all_cards = list(hero_hand) + list(board)
    suit_counts: Counter = Counter(c.suit for c in all_cards)
    hero_suits = {hero_hand[0].suit, hero_hand[1].suit}

    # Backdoor flush: 3 cards of same suit including at least 1 from hero
    for suit, count in suit_counts.items():
        if count == 3 and suit in hero_suits:
            return True

    # Backdoor straight: 3 consecutive ranks including a hero card
    hero_ranks = {int(c.rank) for c in hero_hand}
    all_ranks = sorted({int(c.rank) for c in all_cards})
    for i in range(len(all_ranks) - 1):
        for j in range(i + 1, len(all_ranks)):
            r1, r2 = all_ranks[i], all_ranks[j]
            if r2 - r1 <= 3:  # within a 4-card window
                if hero_ranks & {r1, r2}:
                    return True

    return False


# ---------------------------------------------------------------------------
# Blocker value
# ---------------------------------------------------------------------------

def calculate_blocker_value(
    hero_hand: Tuple[Card, Card], board: List[Card]
) -> Dict[str, float]:
    """Return a dict describing what strong villain combos hero blocks."""
    result: Dict[str, float] = {
        "blocks_top_pair": 0.0,
        "blocks_nut_flush": 0.0,
        "blocks_straights": 0.0,
        "blocks_sets": 0.0,
        "blocks_two_pair": 0.0,
    }
    if not board:
        return result

    board_ranks = [c.rank for c in board]
    board_suits = Counter(c.suit for c in board)
    top_rank = max(board_ranks)
    hero_ranks = {hero_hand[0].rank, hero_hand[1].rank}
    hero_suits = {hero_hand[0].suit, hero_hand[1].suit}

    # Blocks top pair: hero holds the top board rank
    top_rank_count = sum(1 for c in hero_hand if c.rank == top_rank)
    result["blocks_top_pair"] = min(1.0, top_rank_count / 2.0)

    # Blocks nut flush: hero has the ace of the flush suit
    flush_suit = None
    for suit, count in board_suits.items():
        if count >= 2:
            flush_suit = suit
            break
    if flush_suit is not None:
        for c in hero_hand:
            if c.suit == flush_suit and c.rank == Rank.ACE:
                result["blocks_nut_flush"] = 1.0
            elif c.suit == flush_suit and c.rank >= Rank.KING:
                result["blocks_nut_flush"] = max(result["blocks_nut_flush"], 0.5)

    # Blocks sets: hero holds a rank that appears on board
    set_block = 0.0
    for c in hero_hand:
        if c.rank in board_ranks:
            set_block += 0.5
    result["blocks_sets"] = min(1.0, set_block)

    # Blocks straights: simple heuristic based on connectivity
    board_ranks_set = set(int(r) for r in board_ranks)
    hero_rank_ints = {int(r) for r in hero_ranks}
    straight_block = 0.0
    for hr in hero_rank_ints:
        for br in board_ranks_set:
            if abs(hr - br) <= 4:
                straight_block += 0.1
    result["blocks_straights"] = min(1.0, straight_block)

    # Blocks two pair: hero holds second board rank
    if len(board_ranks) >= 2:
        second_rank = sorted(board_ranks, reverse=True)[1]
        for c in hero_hand:
            if c.rank == second_rank:
                result["blocks_two_pair"] += 0.5
        result["blocks_two_pair"] = min(1.0, result["blocks_two_pair"])

    return result


# ---------------------------------------------------------------------------
# Main classification
# ---------------------------------------------------------------------------

def classify_hand(hero_hand: Tuple[Card, Card], board: List[Card]) -> HandStrength:
    """Classify hero's hand strength and draw type on the current board."""
    draw = detect_draws(hero_hand, board)

    if len(board) < 3:
        # Preflop / not enough board cards
        h1, h2 = hero_hand
        r1, r2 = max(h1.rank, h2.rank), min(h1.rank, h2.rank)
        if r1 == r2:
            if r1 >= Rank.QUEEN:
                return HandStrength(MadeHandType.OVERPAIR_BIG, DrawType.NONE, 0.80, False, True)
            return HandStrength(MadeHandType.OVERPAIR_SMALL, DrawType.NONE, 0.70, False, True)
        if r1 == Rank.ACE:
            return HandStrength(MadeHandType.ACE_HIGH, DrawType.NONE, 0.55, False, False)
        return HandStrength(MadeHandType.NO_PAIR, DrawType.NONE, 0.30, False, False)

    all_cards = list(hero_hand) + list(board)
    hand_rank, tiebreakers = evaluate_7(all_cards)
    board_ranks = sorted([c.rank for c in board], reverse=True)
    h1, h2 = hero_hand
    r1 = max(h1.rank, h2.rank)
    r2 = min(h1.rank, h2.rank)

    if hand_rank == HandRank.STRAIGHT_FLUSH:
        return HandStrength(MadeHandType.STRAIGHT_FLUSH, DrawType.NONE, 1.0, False, True)

    if hand_rank == HandRank.FOUR_OF_A_KIND:
        return HandStrength(MadeHandType.QUADS, DrawType.NONE, 0.99, False, True)

    if hand_rank == HandRank.FULL_HOUSE:
        return HandStrength(MadeHandType.FULL_HOUSE, DrawType.NONE, 0.97, False, True)

    if hand_rank == HandRank.FLUSH:
        if is_nut_hand(hero_hand, board):
            return HandStrength(MadeHandType.FLUSH_NUT, DrawType.NONE, 0.95, False, True)
        return HandStrength(MadeHandType.FLUSH_LOW, DrawType.NONE, 0.84, False, True)

    if hand_rank == HandRank.STRAIGHT:
        if is_nut_hand(hero_hand, board):
            return HandStrength(MadeHandType.STRAIGHT_NUT, DrawType.NONE, 0.90, False, True)
        return HandStrength(MadeHandType.STRAIGHT_NON_NUT, DrawType.NONE, 0.78, False, True)

    if hand_rank == HandRank.THREE_OF_A_KIND:
        hero_rank_set = {h1.rank, h2.rank}
        board_rank_set = {c.rank for c in board}
        board_rank_counts = Counter(c.rank for c in board)
        if h1.rank == h2.rank and h1.rank in board_rank_set:
            # Set (pocket pair + board card)
            return HandStrength(MadeHandType.TRIPS_SET, draw, 0.75, False, True)
        elif board_rank_counts.get(tiebreakers[0], 0) >= 3:
            # Board trips, hero uses them
            return HandStrength(MadeHandType.TRIPS_BOARD, draw, 0.60, False, True)
        else:
            # Set where board has two matching hero cards
            return HandStrength(MadeHandType.TRIPS_SET, draw, 0.75, False, True)

    if hand_rank == HandRank.TWO_PAIR:
        # Determine if it's top two or bottom two
        pair_ranks = sorted(tiebreakers[:2], reverse=True)
        if len(board_ranks) >= 1 and pair_ranks[0] >= board_ranks[0]:
            return HandStrength(MadeHandType.TOP_TWO_PAIR, draw, 0.62, False, True)
        return HandStrength(MadeHandType.BOTTOM_TWO_PAIR, draw, 0.48, True, True)

    if hand_rank == HandRank.ONE_PAIR:
        pair_rank = Rank(tiebreakers[0])
        hero_has_pair = (h1.rank == pair_rank or h2.rank == pair_rank)

        # Check for overpair (pocket pair beats all board cards)
        if h1.rank == h2.rank:
            board_max = max(board_ranks)
            if h1.rank > board_max:
                if h1.rank >= Rank.QUEEN:
                    return HandStrength(MadeHandType.OVERPAIR_BIG, draw, 0.70, True, True)
                return HandStrength(MadeHandType.OVERPAIR_SMALL, draw, 0.60, True, True)

        # Hero has top pair?
        if hero_has_pair and board_ranks and pair_rank == board_ranks[0]:
            kicker = r2 if h1.rank == pair_rank or h2.rank == pair_rank else r1
            # kicker is whichever hero card is NOT the paired rank
            if h1.rank == pair_rank:
                kicker = h2.rank
            else:
                kicker = h1.rank
            if kicker >= Rank.KING:
                return HandStrength(MadeHandType.TOP_PAIR_TOP_KICKER, draw, 0.65, False, True)
            if kicker >= Rank.TEN:
                return HandStrength(MadeHandType.TOP_PAIR_GOOD_KICKER, draw, 0.58, False, True)
            return HandStrength(MadeHandType.TOP_PAIR_WEAK_KICKER, draw, 0.50, True, True)

        # Middle pair
        if hero_has_pair and len(board_ranks) >= 2 and pair_rank == board_ranks[1]:
            return HandStrength(MadeHandType.MIDDLE_PAIR, draw, 0.42, True, True)

        # Weak middle pair
        if hero_has_pair and len(board_ranks) >= 2:
            return HandStrength(MadeHandType.WEAK_MIDDLE_PAIR, draw, 0.38, True, True)

        # Bottom pair
        if hero_has_pair:
            return HandStrength(MadeHandType.BOTTOM_PAIR, draw, 0.32, True, False)

        # Pair is on the board and hero doesn't contribute to it
        if not hero_has_pair:
            return HandStrength(MadeHandType.WEAK_MIDDLE_PAIR, draw, 0.35, True, False)

        return HandStrength(MadeHandType.BOTTOM_PAIR, draw, 0.30, True, False)

    # HIGH_CARD
    if h1.rank == Rank.ACE or h2.rank == Rank.ACE:
        return HandStrength(MadeHandType.ACE_HIGH, draw, 0.18, False, False)
    return HandStrength(MadeHandType.NO_PAIR, draw, 0.08, False, False)
