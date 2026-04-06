"""Tests for GTO mixed strategy (src/mixed_strategy.py)."""

import pytest
from collections import Counter
from src.mixed_strategy import mixed_decision


def test_large_ev_gap_deterministic():
    """When EV gap >= 0.15, always pick the highest EV action."""
    # bet EV is much better than check
    for _ in range(20):
        action, conf = mixed_decision(ev_bet=1.0, ev_check=0.5)
        assert action == "bet", f"Expected bet for large gap, got {action}"
    assert conf >= 0.8


def test_large_ev_gap_fold_wins():
    """When fold has best EV (gap >= 0.15), always fold."""
    for _ in range(10):
        action, _ = mixed_decision(
            ev_bet=-0.5, ev_check=-0.3, ev_fold=0.0, ev_call=-0.2
        )
        assert action == "fold"


def test_small_ev_gap_mixed():
    """When EV gap < 0.15, actions should be mixed (not always the same)."""
    results = Counter()
    for _ in range(200):
        action, _ = mixed_decision(ev_bet=0.10, ev_check=0.08, temperature=1.0)
        results[action] += 1
    # Both actions should appear
    assert results["bet"] > 0, "bet should appear in mixed strategy"
    assert results["check"] > 0, "check should appear in mixed strategy"


def test_temperature_zero_pure_strategy():
    """Temperature=0 should always pick the highest EV action."""
    for _ in range(20):
        action, _ = mixed_decision(ev_bet=0.10, ev_check=0.09, temperature=0)
        assert action == "bet"


def test_temperature_high_more_random():
    """Higher temperature should produce more uniform distribution."""
    n = 200
    results_low = Counter()
    results_high = Counter()
    for _ in range(n):
        action, _ = mixed_decision(ev_bet=0.10, ev_check=0.09, temperature=0.5)
        results_low[action] += 1
    for _ in range(n):
        action, _ = mixed_decision(ev_bet=0.10, ev_check=0.09, temperature=5.0)
        results_high[action] += 1
    # Higher temperature → check appears more often
    check_ratio_low = results_low["check"] / n
    check_ratio_high = results_high["check"] / n
    assert check_ratio_high >= check_ratio_low


def test_returns_tuple():
    """mixed_decision should return (action_str, confidence_float)."""
    result = mixed_decision(ev_bet=0.5, ev_check=0.3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    action, conf = result
    assert isinstance(action, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_no_actions_returns_check():
    """When no EV values are provided, defaults to check with 0.5 confidence."""
    action, conf = mixed_decision()
    assert action == "check"
    assert conf == 0.5


def test_single_action_always_chosen():
    """With only one available action, it should always be chosen."""
    for _ in range(10):
        action, _ = mixed_decision(ev_bet=0.3)
        assert action == "bet"


def test_all_ev_options():
    """Test with all 5 action types."""
    action, conf = mixed_decision(
        ev_bet=0.5, ev_check=0.3, ev_call=0.2, ev_fold=0.0, ev_raise=0.8
    )
    # raise has highest EV with large gap → deterministic
    assert action == "raise"


def test_confidence_range():
    """Confidence should always be between 0 and 1."""
    for _ in range(50):
        action, conf = mixed_decision(ev_bet=0.10, ev_check=0.08, temperature=2.0)
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


def test_negative_ev_fold_wins():
    """When all actions are negative EV, fold (0.0) should win."""
    # ev_fold = 0 is always 0 (folding gives up pot but no further investment)
    action, _ = mixed_decision(ev_bet=-0.5, ev_check=-0.3, ev_fold=0.0)
    # fold has highest EV; gap = 0.3 >= 0.15 → deterministic
    assert action == "fold"
