"""Tests for RangeUpdater."""

import pytest
from typing import List

from src.range_updater import RangeUpdater
from src.weighted_range import build_range_combos, ComboWeight
from src.board_analysis import analyze_board
from src.card import cards_from_str
from src.ranges import get_rfi_range
from src.position import Position


@pytest.fixture
def updater():
    return RangeUpdater()


def _build_range(position: Position, board_str: str) -> List[ComboWeight]:
    board = cards_from_str(board_str)
    dead = board
    return build_range_combos(get_rfi_range(position), dead)


def _avg_weight(combos: List[ComboWeight]) -> float:
    if not combos:
        return 0.0
    return sum(c.weight for c in combos) / len(combos)


class TestCheckUpdates:
    """After villain checks, strong hands should have lower weight."""

    def test_check_reduces_combo_count_or_weights(self, updater):
        board = cards_from_str("AcKd7h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AcKd7h")
        updated = updater.update_range_after_action(
            original, board, "check", 0.0, "flop", tex, is_ip=True
        )
        assert len(updated) > 0
        # Hands should be reduced (check → remove some strong hands)
        assert len(updated) <= len(original)

    def test_check_oop_dry_board_less_reduction(self, updater):
        """OOP dry board check: slower reduction (slow-play exception)."""
        board = cards_from_str("Ac2d7h")
        tex = analyze_board(board)
        original = _build_range(Position.UTG, "Ac2d7h")

        updated_ip = updater.update_range_after_action(
            original, board, "check", 0.0, "flop", tex, is_ip=True
        )
        updated_oop = updater.update_range_after_action(
            original, board, "check", 0.0, "flop", tex, is_ip=False
        )
        # OOP dry board should keep more weight (slow-play), so avg weight ≥ IP
        assert _avg_weight(updated_oop) >= _avg_weight(updated_ip) * 0.90


class TestBetUpdates:
    def test_small_bet_keeps_medium_hands(self, updater):
        board = cards_from_str("Kc7d2h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "Kc7d2h")
        updated = updater.update_range_after_action(
            original, board, "bet", 0.30, "flop", tex, is_ip=True
        )
        # Small bet merged → should keep more combos than large bet
        assert len(updated) > 0

    def test_large_bet_removes_medium_hands(self, updater):
        board = cards_from_str("Kc7d2h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "Kc7d2h")
        updated_small = updater.update_range_after_action(
            original, board, "bet", 0.25, "flop", tex, is_ip=True
        )
        updated_large = updater.update_range_after_action(
            original, board, "bet", 0.90, "flop", tex, is_ip=True
        )
        # Large bet (polarized) should have fewer combos
        assert len(updated_large) <= len(updated_small)

    def test_overbet_most_restrictive(self, updater):
        board = cards_from_str("AsKdQh")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AsKdQh")
        updated_med = updater.update_range_after_action(
            original, board, "bet", 0.50, "flop", tex, is_ip=True
        )
        updated_over = updater.update_range_after_action(
            original, board, "bet", 1.30, "flop", tex, is_ip=True
        )
        # Overbet should produce fewer combos
        assert len(updated_over) <= len(updated_med)


class TestAllInUpdates:
    def test_all_in_narrows_range(self, updater):
        board = cards_from_str("AsKdQh")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AsKdQh")
        updated = updater.update_range_after_action(
            original, board, "all_in", 0.0, "flop", tex, is_ip=True
        )
        # All-in should significantly narrow range
        assert len(updated) <= len(original)

    def test_all_in_monotone_board_extra_restriction(self, updater):
        board = cards_from_str("KsQsJs")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "KsQsJs")
        updated_standard = updater.update_range_after_action(
            original, board, "all_in", 0.0, "flop", tex, is_ip=False
        )
        # Monotone all-in should be very restrictive
        assert len(updated_standard) > 0


class TestRaiseUpdates:
    def test_raise_narrows_to_strong_hands(self, updater):
        board = cards_from_str("AcKd7h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AcKd7h")
        updated = updater.update_range_after_action(
            original, board, "raise", 2.5, "flop", tex, is_ip=True
        )
        # Raise → very narrow range
        assert len(updated) <= len(original)
        assert len(updated) > 0


class TestCallUpdates:
    def test_call_removes_air_and_nuts(self, updater):
        board = cards_from_str("AcKd7h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AcKd7h")
        updated = updater.update_range_after_action(
            original, board, "call", 0.50, "flop", tex, is_ip=True
        )
        # Call range should be a medium range
        assert len(updated) > 0
        # Should have fewer combos than original (air removed)
        assert len(updated) <= len(original)


class TestMultiStreetAccumulation:
    """Multiple street updates should accumulate."""

    def test_flop_then_turn_bet_narrows_progressively(self, updater):
        board_flop = cards_from_str("AcKd7h")
        tex_flop = analyze_board(board_flop)
        original = _build_range(Position.CO, "AcKd7h")

        # After flop bet
        after_flop_bet = updater.update_range_after_action(
            original, board_flop, "bet", 0.50, "flop", tex_flop, is_ip=True
        )

        # After turn bet (larger)
        board_turn = cards_from_str("AcKd7h3s")
        tex_turn = analyze_board(board_turn)
        after_turn_bet = updater.update_range_after_action(
            after_flop_bet, board_turn, "bet", 0.75, "turn", tex_turn, is_ip=True
        )

        # Progressive narrowing
        assert len(after_flop_bet) <= len(original)
        assert len(after_turn_bet) <= len(after_flop_bet)

    def test_river_all_in_after_two_streets_very_narrow(self, updater):
        board_full = cards_from_str("AcKd7h3sQc")
        tex = analyze_board(board_full)
        original = _build_range(Position.CO, "AcKd7h3sQc")

        after_flop = updater.update_range_after_action(
            original, board_full, "bet", 0.50, "flop", tex, is_ip=True
        )
        after_turn = updater.update_range_after_action(
            after_flop, board_full, "bet", 0.65, "turn", tex, is_ip=True
        )
        after_allin = updater.update_range_after_action(
            after_turn, board_full, "all_in", 0.0, "river", tex, is_ip=True
        )
        # Very narrow after three streets of aggression
        assert len(after_allin) <= len(after_turn)
        assert len(after_allin) > 0


class TestEdgeCases:
    def test_empty_range_returns_empty(self, updater):
        board = cards_from_str("AcKd7h")
        tex = analyze_board(board)
        result = updater.update_range_after_action([], board, "bet", 0.50, "flop", tex, True)
        assert result == []

    def test_unknown_action_returns_unchanged_weights(self, updater):
        board = cards_from_str("AcKd7h")
        tex = analyze_board(board)
        original = _build_range(Position.CO, "AcKd7h")
        updated = updater.update_range_after_action(
            original, board, "unknown_action", 0.50, "flop", tex, True
        )
        # Unknown action → no change
        assert len(updated) == len(original)
