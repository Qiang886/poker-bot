"""Tests for SizingTellInterpreter."""

import pytest

from src.sizing_tell import SizingTellInterpreter, SizingInterpretation
from src.board_analysis import analyze_board
from src.card import cards_from_str
from src.opponent import VillainProfile


@pytest.fixture
def interpreter():
    return SizingTellInterpreter()


def _dry_board_texture():
    return analyze_board(cards_from_str("Ac7d2h"))


def _monotone_board_texture():
    return analyze_board(cards_from_str("KsQsJs"))


def _wet_board_texture():
    return analyze_board(cards_from_str("9s8s7h"))


class TestBasicSizing:
    def test_small_bet_is_merged(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 0.30, "flop", tex)
        assert result.polarization == "merged"
        assert result.estimated_bluff_pct < 0.20

    def test_medium_bet_is_standard(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 0.50, "flop", tex)
        assert result.polarization == "standard"
        assert 0.20 <= result.estimated_bluff_pct <= 0.35

    def test_large_bet_is_polarized(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 0.80, "flop", tex)
        assert result.polarization == "polarized"
        assert result.estimated_bluff_pct >= 0.25

    def test_overbet_is_very_polarized(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 1.20, "flop", tex)
        assert result.polarization == "very_polarized"
        assert result.estimated_bluff_pct >= 0.30

    def test_river_overbet_very_polarized(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 1.50, "river", tex)
        assert "very_polarized" in result.polarization
        assert result.estimated_bluff_pct >= 0.30


class TestAllIn:
    def test_all_in_standard_board(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("all_in", 0.0, "turn", tex)
        assert result.polarization == "very_polarized"
        assert result.estimated_value_pct >= 0.55

    def test_all_in_monotone_board_value_heavy(self, interpreter):
        tex = _monotone_board_texture()
        result = interpreter.interpret("all_in", 0.0, "flop", tex)
        # Monotone board + all-in: very value-heavy
        assert result.estimated_value_pct >= 0.70
        assert result.estimated_bluff_pct <= 0.15
        assert "monotone" in result.description.lower()

    def test_all_in_wet_board(self, interpreter):
        tex = _wet_board_texture()
        result = interpreter.interpret("all_in", 0.0, "flop", tex)
        assert result.estimated_value_pct >= 0.60
        # More bluffs allowed on wet board (draws)
        assert result.estimated_bluff_pct >= 0.12


class TestBoardAdjustments:
    def test_monotone_board_large_bet_value_heavy(self, interpreter):
        tex = _monotone_board_texture()
        result = interpreter.interpret("bet", 0.80, "flop", tex)
        # Monotone board + large bet → more value, fewer bluffs
        assert result.estimated_value_pct > 0.55
        assert "monotone" in result.description.lower()

    def test_dry_board_small_bet_range_cbet(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 0.25, "flop", tex)
        # Dry board small bet → range cbet note in description
        assert "range cbet" in result.description.lower() or result.estimated_bluff_pct >= 0.12


class TestVillainProfileAdjustment:
    def test_passive_villain_mostly_value(self, interpreter):
        tex = _dry_board_texture()
        vp = VillainProfile()
        vp.stats.aggression_factor = 1.0
        vp.stats.hands_played = 150
        result = interpreter.interpret("bet", 0.60, "flop", tex, villain_profile=vp)
        base = interpreter.interpret("bet", 0.60, "flop", tex)
        # Passive villain → fewer bluffs
        assert result.estimated_bluff_pct <= base.estimated_bluff_pct

    def test_lag_villain_more_bluffs(self, interpreter):
        tex = _dry_board_texture()
        vp = VillainProfile()
        vp.stats.aggression_factor = 4.0
        vp.stats.hands_played = 150
        result = interpreter.interpret("bet", 0.60, "flop", tex, villain_profile=vp)
        base = interpreter.interpret("bet", 0.60, "flop", tex)
        # LAG villain → more bluffs
        assert result.estimated_bluff_pct >= base.estimated_bluff_pct

    def test_insufficient_sample_no_profile_adjustment(self, interpreter):
        tex = _dry_board_texture()
        vp = VillainProfile()
        vp.stats.aggression_factor = 5.0
        vp.stats.hands_played = 10  # not enough sample
        result_with_profile = interpreter.interpret("bet", 0.60, "flop", tex, villain_profile=vp)
        result_no_profile = interpreter.interpret("bet", 0.60, "flop", tex)
        # Profile ignored (< 100 hands)
        assert result_with_profile.estimated_bluff_pct == result_no_profile.estimated_bluff_pct


class TestReturnType:
    def test_returns_sizing_interpretation(self, interpreter):
        tex = _dry_board_texture()
        result = interpreter.interpret("bet", 0.50, "flop", tex)
        assert isinstance(result, SizingInterpretation)
        assert isinstance(result.description, str)
        assert 0.0 <= result.estimated_value_pct <= 1.0
        assert 0.0 <= result.estimated_bluff_pct <= 1.0
