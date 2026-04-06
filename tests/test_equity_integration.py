"""Integration tests: calculate_equity() wired into PostflopEngine.decide()."""

from unittest.mock import patch

import pytest

from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.weighted_range import build_range_combos, ComboWeight
from src.ranges import get_rfi_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine():
    return PostflopEngine()


def _decide(
    hand_str,
    board_str,
    street="flop",
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    position=Position.BTN,
    villain_range=None,
    is_multiway=False,
    sizing_ratio=0.0,
):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    if villain_range is None:
        villain_range = build_range_combos(get_rfi_range(Position.UTG), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    return engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=position,
        street=street,
        villain_profile=None,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=is_multiway,
        sizing_ratio=sizing_ratio,
    )


def _get_num_simulations(mock_call):
    """Extract num_simulations from a mock call, accepting positional or keyword arg."""
    args, kwargs = mock_call
    if "num_simulations" in kwargs:
        return kwargs["num_simulations"]
    # calculate_equity(hero_hand, villain_range, board, num_simulations)
    if len(args) > 3:
        return args[3]
    return None


# ---------------------------------------------------------------------------
# 1. calculate_equity() is called when villain_range is non-empty
# ---------------------------------------------------------------------------

class TestCalculateEquityCalled:
    def test_called_on_flop_with_villain_range(self):
        """calculate_equity should be called exactly once on a flop decision."""
        with patch("src.postflop.calculate_equity", wraps=__import__(
            "src.equity", fromlist=["calculate_equity"]
        ).calculate_equity) as mock_eq:
            _decide("AhKc", "As2d7h", street="flop")
            assert mock_eq.call_count == 1

    def test_called_on_turn_with_villain_range(self):
        with patch("src.postflop.calculate_equity", wraps=__import__(
            "src.equity", fromlist=["calculate_equity"]
        ).calculate_equity) as mock_eq:
            _decide("AhKc", "As2d7h3c", street="turn")
            assert mock_eq.call_count == 1

    def test_called_on_river_with_villain_range(self):
        with patch("src.postflop.calculate_equity", wraps=__import__(
            "src.equity", fromlist=["calculate_equity"]
        ).calculate_equity) as mock_eq:
            _decide("AhKc", "As2d7h3c9s", street="river")
            assert mock_eq.call_count == 1

    def test_flop_uses_500_simulations(self):
        with patch("src.postflop.calculate_equity") as mock_eq:
            mock_eq.return_value = 0.55
            _decide("AhKc", "As2d7h", street="flop")
            assert _get_num_simulations(mock_eq.call_args) == 500

    def test_turn_uses_500_simulations(self):
        with patch("src.postflop.calculate_equity") as mock_eq:
            mock_eq.return_value = 0.55
            _decide("AhKc", "As2d7h3c", street="turn")
            assert _get_num_simulations(mock_eq.call_args) == 500

    def test_river_uses_300_simulations(self):
        with patch("src.postflop.calculate_equity") as mock_eq:
            mock_eq.return_value = 0.55
            _decide("AhKc", "As2d7h3c9s", street="river")
            assert _get_num_simulations(mock_eq.call_args) == 300


# ---------------------------------------------------------------------------
# 2. Fallback to equity_bucket when villain_range is empty
# ---------------------------------------------------------------------------

class TestFallbackWhenNoVillainRange:
    def test_not_called_when_villain_range_empty(self):
        """calculate_equity must NOT be called if villain_range is empty."""
        with patch("src.postflop.calculate_equity") as mock_eq:
            _decide("AhKc", "As2d7h", street="flop", villain_range=[])
            mock_eq.assert_not_called()

    def test_decision_still_made_with_empty_range(self):
        """A valid decision should be returned even with no villain range."""
        d = _decide("AhKc", "As2d7h", street="flop", villain_range=[])
        assert d.action in ("bet", "check", "call", "fold", "raise", "all-in")


# ---------------------------------------------------------------------------
# 3. _facing_bet_decision uses real equity for EV calculation
# ---------------------------------------------------------------------------

class TestFacingBetEquityEV:
    def test_high_equity_calls_bet(self):
        """With very high real equity (mocked ~0.80), hero should call/raise a bet."""
        with patch("src.postflop.calculate_equity", return_value=0.80):
            # Middle pair facing a small bet; high real equity should call
            d = _decide("7h7c", "7s2cKd", street="flop", to_call=3.0, pot=10.0)
            assert d.action in ("call", "raise", "all-in")

    def test_low_equity_folds_to_bet(self):
        """With very low real equity (mocked ~0.10), hero should fold to a bet."""
        with patch("src.postflop.calculate_equity", return_value=0.10):
            # Even TPTK, with 10% equity (e.g. dominated), should fold
            d = _decide("2h3d", "AsKhQc", street="flop", to_call=8.0, pot=10.0)
            assert d.action == "fold"

    def test_ev_positive_leads_to_call(self):
        """EV = real_equity * (pot + to_call) - to_call > 0 should produce a call."""
        # equity=0.65, pot=10, to_call=4 → EV = 0.65*14 - 4 = 5.1 > 0
        with patch("src.postflop.calculate_equity", return_value=0.65):
            d = _decide("AhKc", "As2d7h", street="flop", to_call=4.0, pot=10.0)
            assert d.action in ("call", "raise", "all-in")

    def test_ev_negative_leads_to_fold(self):
        """EV < 0 with weak hand should fold."""
        # equity=0.15, pot=10, to_call=8 → EV = 0.15*18 - 8 = -5.3 < 0
        with patch("src.postflop.calculate_equity", return_value=0.15):
            d = _decide("2h3d", "AsKhQc", street="flop", to_call=8.0, pot=10.0)
            assert d.action == "fold"


# ---------------------------------------------------------------------------
# 4. _first_to_act_decision uses real equity for bet/check
# ---------------------------------------------------------------------------

class TestFirstToActEquity:
    def test_high_equity_bets(self):
        """real_equity > 0.6 should produce a bet (value bet) when first to act."""
        with patch("src.postflop.calculate_equity", return_value=0.75):
            d = _decide("AhKc", "As2d7h", street="flop", to_call=0.0)
            assert d.action in ("bet", "raise", "all-in")

    def test_low_equity_checks_no_draw(self):
        """real_equity < 0.35 without draw should not value-bet."""
        with patch("src.postflop.calculate_equity", return_value=0.20):
            # Weak hand with no draw → check
            d = _decide("2h3d", "AsKhQc", street="flop", to_call=0.0)
            assert d.action == "check"

    def test_medium_equity_checks_showdown_value(self):
        """0.35 < real_equity < 0.6 should lean toward check (showdown value)."""
        with patch("src.postflop.calculate_equity", return_value=0.48):
            # Weak two-card hand on a dry board → check-back
            d = _decide("2h3d", "AsKhQc", street="flop", to_call=0.0)
            assert d.action == "check"


# ---------------------------------------------------------------------------
# 5. River equity is precise (5 board cards)
# ---------------------------------------------------------------------------

class TestRiverEquityPrecision:
    def test_strong_river_hand_bets(self):
        """On the river, a strong hand should result in a value bet."""
        d = _decide("AhKc", "As2d7h3c9s", street="river", to_call=0.0)
        assert d.action in ("bet", "raise", "all-in")

    def test_river_equity_consistent(self):
        """River equity for nut hand should be high (>0.6) on multiple runs."""
        from src.equity import calculate_equity
        hand = tuple(cards_from_str("AhKc"))
        board = cards_from_str("As2d7h3c9s")
        dead = list(hand) + board
        vr = build_range_combos(get_rfi_range(Position.UTG), dead)
        equities = [calculate_equity(hand, vr, board, num_simulations=300) for _ in range(5)]
        avg = sum(equities) / len(equities)
        assert avg > 0.55, f"Expected river equity > 0.55 for TPTK, got {avg:.3f}"

    def test_river_facing_bet_calls_with_real_equity(self):
        """River bluff-catch: with high real equity, should call or raise."""
        with patch("src.postflop.calculate_equity", return_value=0.72):
            d = _decide("AhKc", "As2d7h3c9s", street="river", to_call=5.0, pot=10.0)
            assert d.action in ("call", "raise", "all-in")

    def test_river_facing_bet_folds_low_equity(self):
        """River facing bet: very low equity should fold."""
        with patch("src.postflop.calculate_equity", return_value=0.08):
            # 2h3d has no pair/draw on this board; weak hand with 0.08 equity → fold
            d = _decide("2h3d", "AsKhQc7d9s", street="river", to_call=8.0, pot=10.0)
            assert d.action == "fold"


# ---------------------------------------------------------------------------
# 6. Turn equity integration
# ---------------------------------------------------------------------------

class TestTurnEquityIntegration:
    def test_high_equity_turn_bets(self):
        """Turn first-to-act with high real equity should bet."""
        with patch("src.postflop.calculate_equity", return_value=0.72):
            d = _decide("AhKc", "As2d7h3c", street="turn", to_call=0.0)
            assert d.action in ("bet", "raise", "all-in")

    def test_low_equity_turn_checks(self):
        """Turn first-to-act with low equity and no draw should check."""
        with patch("src.postflop.calculate_equity", return_value=0.18):
            d = _decide("2h3d", "AsKhQc8s", street="turn", to_call=0.0)
            assert d.action == "check"

    def test_positive_ev_turn_facing_bet_calls(self):
        """Turn facing bet with positive EV should call."""
        # equity=0.65, pot=10, to_call=3 → EV = 0.65*13 - 3 = 5.45 > 0
        with patch("src.postflop.calculate_equity", return_value=0.65):
            d = _decide("AhKc", "As2d7h3c", street="turn", to_call=3.0, pot=10.0)
            assert d.action in ("call", "raise", "all-in")
