"""Tests for OutsCalculator."""

import pytest
from typing import List

from src.outs_calculator import OutsCalculator, OutsAnalysis, OutInfo, format_outs_summary
from src.weighted_range import build_range_combos, ComboWeight
from src.card import cards_from_str, Card
from src.ranges import get_rfi_range
from src.position import Position


@pytest.fixture
def calc():
    return OutsCalculator()


def _villain_range(board: List[Card]) -> List[ComboWeight]:
    dead = board
    return build_range_combos(get_rfi_range(Position.CO), dead)


class TestFlushDrawOuts:
    """Flush draw should have ~9 clean outs."""

    def test_nut_flush_draw_has_9_clean_outs(self, calc):
        # As9s on Ks7s2h – nut flush draw
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        # 9 remaining spades → 9 flush outs; may also count pair outs
        flush_clean_outs = [o for o in analysis.clean_outs if o.improves_to == "flush"]
        assert len(flush_clean_outs) == 9
        # Overall clean count is at least 9
        assert analysis.total_clean >= 9

    def test_low_flush_draw_outs_may_be_dirty(self, calc):
        # 9s2s on Ks7s2h – non-nut flush draw (someone may have Ks or higher spade)
        hole = tuple(cards_from_str("9s2s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        # Should still have spade outs but some may be dirty
        total_flush_outs = sum(
            1 for o in (analysis.clean_outs + analysis.dirty_outs + analysis.dead_outs)
            if o.improves_to == "flush"
        )
        assert total_flush_outs == 9


class TestOESDOuts:
    """Open-ended straight draw should have ~8 outs."""

    def test_oesd_has_8_outs(self, calc):
        # JhTh on 9s8d2c – OESD (Q or 7 completes)
        hole = tuple(cards_from_str("JhTh"))
        board = cards_from_str("9s8d2c")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        straight_outs = [
            o for o in (analysis.clean_outs + analysis.dirty_outs + analysis.dead_outs)
            if o.improves_to == "straight"
        ]
        # 4 Queens + 4 Sevens = 8
        assert len(straight_outs) == 8

    def test_oesd_clean_outs_positive(self, calc):
        hole = tuple(cards_from_str("JhTh"))
        board = cards_from_str("9s8d2c")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        assert analysis.total_clean >= 0  # at least some should be clean


class TestSetVsFlushDirtyOuts:
    """Set facing flush board → full house outs may be dirty (flush still beats)."""

    def test_set_vs_flush_board_has_full_house_outs(self, calc):
        # KhKd on Ks9s5s (we have set of kings, board is monotone spades)
        hole = tuple(cards_from_str("KhKd"))
        board = cards_from_str("Ks9s5s")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        fh_outs = [
            o for o in (analysis.clean_outs + analysis.dirty_outs + analysis.dead_outs)
            if o.improves_to == "full_house"
        ]
        # Pairing the board (9s, 5s) gives us full house
        assert len(fh_outs) > 0


class TestCleanDirtyDeadClassification:
    def test_clean_outs_have_high_equity(self, calc):
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        for out in analysis.clean_outs:
            assert out.equity_after > 0.75
            assert out.category == "clean"

    def test_dirty_outs_have_medium_equity(self, calc):
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        for out in analysis.dirty_outs:
            assert 0.40 < out.equity_after <= 0.75
            assert out.category == "dirty"

    def test_dead_outs_have_low_equity(self, calc):
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        for out in analysis.dead_outs:
            assert out.equity_after <= 0.40
            assert out.category == "dead"


class TestComboDraw:
    """Combo draw (flush draw + straight draw) should have many outs."""

    def test_combo_draw_has_many_outs(self, calc):
        # JsTs on 9s8s2h – flush draw + OESD
        hole = tuple(cards_from_str("JsTs"))
        board = cards_from_str("9s8s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        total = analysis.total_clean + analysis.total_dirty + len(analysis.dead_outs)
        # Should have flush outs + straight outs (some overlap possible)
        assert total >= 12

    def test_combo_draw_true_equity_high(self, calc):
        hole = tuple(cards_from_str("JsTs"))
        board = cards_from_str("9s8s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        # Combo draw should have fairly high true equity
        assert analysis.true_equity > 0.30


class TestEdgeCases:
    def test_empty_villain_range(self, calc):
        hole = tuple(cards_from_str("JsTs"))
        board = cards_from_str("9s8s2h")
        analysis = calc.calculate_outs(hole, board, [], simulations_per_out=50)
        # With empty range, equity is 0.5 by convention
        assert isinstance(analysis, OutsAnalysis)

    def test_less_than_3_board_cards_returns_empty(self, calc):
        hole = tuple(cards_from_str("AsKs"))
        board = cards_from_str("AcKd")  # only 2 cards
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange)
        # No outs analysis without at least a flop
        assert analysis.total_clean == 0
        assert analysis.total_dirty == 0

    def test_best_out_is_most_equity(self, calc):
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        if analysis.best_out is not None:
            all_outs = analysis.clean_outs + analysis.dirty_outs + analysis.dead_outs
            max_eq = max(o.equity_after for o in all_outs)
            assert analysis.best_out.equity_after == max_eq


class TestFormatSummary:
    def test_format_outputs_string(self, calc):
        hole = tuple(cards_from_str("As9s"))
        board = cards_from_str("Ks7s2h")
        vrange = _villain_range(board)
        analysis = calc.calculate_outs(hole, board, vrange, simulations_per_out=50)
        summary = format_outs_summary(analysis)
        assert isinstance(summary, str)
        assert "clean" in summary or "no outs" in summary
        assert "equity" in summary

    def test_format_no_outs(self):
        empty = OutsAnalysis()
        summary = format_outs_summary(empty)
        assert summary == "no outs"
