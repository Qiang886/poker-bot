"""Precise outs calculation: classify outs as clean, dirty, or dead."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.card import Card, FULL_DECK
from src.evaluator import evaluate_7, HandRank
from src.hand_analysis import classify_hand, MadeHandType
from src.equity import calculate_equity
from src.weighted_range import ComboWeight


@dataclass
class OutInfo:
    card: Card                  # the out card
    improves_to: str            # "flush", "straight", "full_house", "trips", "two_pair", "pair"
    is_nut: bool                # whether the improved hand is likely the nuts
    equity_after: float         # equity vs villain range after hitting this out
    category: str               # "clean", "dirty", "dead"
    danger: str                 # "none", "flush_possible", "straight_possible", "board_pairs"


@dataclass
class OutsAnalysis:
    clean_outs: List[OutInfo] = field(default_factory=list)   # equity_after > 0.75
    dirty_outs: List[OutInfo] = field(default_factory=list)   # 0.40 < equity_after <= 0.75
    dead_outs: List[OutInfo] = field(default_factory=list)    # equity_after <= 0.40
    total_clean: int = 0
    total_dirty: int = 0
    true_equity: float = 0.0    # probability-weighted equity across all outs
    best_out: Optional[OutInfo] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _made_hand_label(rank: HandRank) -> str:
    label_map = {
        HandRank.HIGH_CARD: "high_card",
        HandRank.ONE_PAIR: "pair",
        HandRank.TWO_PAIR: "two_pair",
        HandRank.THREE_OF_A_KIND: "trips",
        HandRank.STRAIGHT: "straight",
        HandRank.FLUSH: "flush",
        HandRank.FULL_HOUSE: "full_house",
        HandRank.FOUR_OF_A_KIND: "quads",
        HandRank.STRAIGHT_FLUSH: "straight_flush",
    }
    return label_map.get(rank, "unknown")


def _improvement_label(before: HandRank, after: HandRank) -> str:
    """Label describing the improvement after the out card."""
    return _made_hand_label(after)

def _is_nut_check(
    hero_hand: Tuple[Card, Card],
    board_with_out: List[Card],
) -> bool:
    """Rough check: is hero's hand likely nut-level on this board?"""
    from src.hand_analysis import is_nut_hand
    try:
        return is_nut_hand(hero_hand, board_with_out)
    except Exception:
        return False


def _detect_danger(board_with_out: List[Card], out_card: Card) -> str:
    """Detect board danger after the out card is added."""
    from collections import Counter

    suits = [c.suit for c in board_with_out]
    suit_counts = Counter(suits)

    # 4+ cards of same suit on board → flush_possible
    if max(suit_counts.values()) >= 4:
        return "flush_possible"

    ranks = [c.rank for c in board_with_out]
    rank_counts = Counter(ranks)
    # Board pairs after out card
    if max(rank_counts.values()) >= 2:
        board_only = [c for c in board_with_out if c != out_card]
        board_ranks_only = Counter(c.rank for c in board_only)
        if rank_counts[out_card.rank] >= 2 or max(rank_counts.values()) >= 3:
            return "board_pairs"

    # Check for open-ended or double-paired situation → straight_possible
    sorted_ranks = sorted(set(int(r) for r in ranks))
    max_consec = 1
    cur = 1
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 1
    if max_consec >= 4:
        return "straight_possible"

    return "none"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutsCalculator:
    """Calculate outs and classify them as clean, dirty, or dead."""

    def calculate_outs(
        self,
        hole: Tuple[Card, Card],
        board: List[Card],
        villain_range: List[ComboWeight],
        simulations_per_out: int = 200,
    ) -> OutsAnalysis:
        """
        Identify all improvement cards in the remaining deck and classify each.

        Returns an OutsAnalysis with clean/dirty/dead outs and true equity.
        """
        if len(board) < 3:
            # Need at least a flop
            return OutsAnalysis()

        dead = set(hole) | set(board)
        remaining = [c for c in FULL_DECK if c not in dead]

        # Evaluate current hand strength
        current_eval = evaluate_7(list(hole) + list(board))
        current_rank = current_eval[0]

        all_outs: List[OutInfo] = []
        # Avoid evaluating same out card twice (shouldn't happen with FULL_DECK but just in case)
        seen_cards = set()

        for candidate in remaining:
            if candidate in seen_cards:
                continue
            seen_cards.add(candidate)

            board_with_out = list(board) + [candidate]
            new_eval = evaluate_7(list(hole) + board_with_out)
            new_rank = new_eval[0]

            # Only count as an out if the HAND CATEGORY improves (not just kicker).
            # HandRank is an IntEnum where higher values = better hands
            # (HIGH_CARD=1 → STRAIGHT_FLUSH=9), so new_rank > current_rank = improvement.
            if new_rank <= current_rank:
                continue

            improves_to = _improvement_label(current_rank, new_rank)
            is_nut = _is_nut_check(hole, board_with_out)
            danger = _detect_danger(board_with_out, candidate)

            # Evaluate equity on board_with_out vs villain range
            # We only go to turn/river scenarios, so board is 3 or 4 cards
            eq = self._equity_after_out(hole, board_with_out, villain_range, simulations_per_out)

            # Classify out
            if eq > 0.75:
                category = "clean"
            elif eq > 0.40:
                category = "dirty"
            else:
                category = "dead"

            all_outs.append(OutInfo(
                card=candidate,
                improves_to=improves_to,
                is_nut=is_nut,
                equity_after=round(eq, 4),
                category=category,
                danger=danger,
            ))

        clean = [o for o in all_outs if o.category == "clean"]
        dirty = [o for o in all_outs if o.category == "dirty"]
        dead_outs = [o for o in all_outs if o.category == "dead"]

        # True equity: average across outs weighted equally
        # Each out card is one of ~(45-|board|) remaining cards
        remaining_count = len(remaining)
        if remaining_count == 0:
            true_eq = 0.0
        elif all_outs:
            # Probability of hitting at least one out on next card
            out_prob = len(all_outs) / remaining_count
            if all_outs:
                avg_eq_if_hit = sum(o.equity_after for o in all_outs) / len(all_outs)
            else:
                avg_eq_if_hit = 0.0
            miss_prob = 1.0 - out_prob
            # If we miss, assume ~20% equity (very rough)
            miss_equity = 0.20
            true_eq = out_prob * avg_eq_if_hit + miss_prob * miss_equity
        else:
            true_eq = 0.20  # no outs → bleak

        best = max(all_outs, key=lambda o: o.equity_after) if all_outs else None

        return OutsAnalysis(
            clean_outs=clean,
            dirty_outs=dirty,
            dead_outs=dead_outs,
            total_clean=len(clean),
            total_dirty=len(dirty),
            true_equity=round(true_eq, 4),
            best_out=best,
        )

    @staticmethod
    def _equity_after_out(
        hole: Tuple[Card, Card],
        board_with_out: List[Card],
        villain_range: List[ComboWeight],
        simulations: int,
    ) -> float:
        """Quick Monte Carlo equity after the out card is on the board."""
        if not villain_range:
            return 0.5
        # Filter villain range: remove combos that conflict with our cards or out card
        dead = set(hole) | set(board_with_out)
        filtered = [
            cw for cw in villain_range
            if cw.combo[0] not in dead and cw.combo[1] not in dead
        ]
        if not filtered:
            return 0.5
        return calculate_equity(hole, filtered, board_with_out, num_simulations=simulations)


def format_outs_summary(analysis: OutsAnalysis) -> str:
    """Human-readable summary of outs analysis."""
    parts = []
    if analysis.total_clean > 0:
        improvements = list({o.improves_to for o in analysis.clean_outs})
        parts.append(f"{analysis.total_clean} clean out(s) ({', '.join(improvements)})")
    if analysis.total_dirty > 0:
        improvements = list({o.improves_to for o in analysis.dirty_outs})
        dangers = list({o.danger for o in analysis.dirty_outs if o.danger != "none"})
        danger_str = f" [{', '.join(dangers)}]" if dangers else ""
        parts.append(f"{analysis.total_dirty} dirty out(s) ({', '.join(improvements)}{danger_str})")
    dead_count = len(analysis.dead_outs)
    if dead_count > 0:
        parts.append(f"{dead_count} dead out(s)")
    if not parts:
        return "no outs"
    return "; ".join(parts) + f" | true equity ≈ {analysis.true_equity:.0%}"
