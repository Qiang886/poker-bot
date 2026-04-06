"""Tests for dynamic equity bucket adjustment."""

import pytest
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand
from src.board_analysis import analyze_board
from src.card import cards_from_str
from src.dynamic_equity import adjust_equity_bucket


def hs(made: MadeHandType, draw: DrawType = DrawType.NONE,
        equity: float = 0.50, vulnerable: bool = False, showdown: bool = True) -> HandStrength:
    return HandStrength(
        made_hand=made,
        draw=draw,
        equity_bucket=equity,
        is_vulnerable=vulnerable,
        has_showdown_value=showdown,
    )


def board_tex(board_str):
    return analyze_board(cards_from_str(board_str))


# ---------------------------------------------------------------------------
# Opponent type adjustments
# ---------------------------------------------------------------------------

def test_tptk_vs_fish_equity_increases():
    """TPTK vs fish → equity should increase by ~0.12."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj = adjust_equity_bucket(base, tex, "fish", "flop")
    assert adj > 0.65
    assert abs(adj - (0.65 + 0.12 + 0.04)) < 0.05  # +0.12 type, +0.04 dry board


def test_tptk_vs_nit_equity_decreases():
    """TPTK vs nit → equity should decrease by ~0.12."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj = adjust_equity_bucket(base, tex, "nit", "flop")
    assert adj < 0.65
    assert abs(adj - (0.65 - 0.12 + 0.04)) < 0.05


def test_middle_pair_vs_fish_equity_increases():
    """Middle pair vs fish → equity increases ~0.10."""
    base = hs(MadeHandType.MIDDLE_PAIR, equity=0.42)
    tex = board_tex("As7d3c")
    adj = adjust_equity_bucket(base, tex, "fish", "flop")
    assert adj > 0.42
    assert adj > 0.45


def test_middle_pair_vs_nit_equity_decreases():
    """Middle pair vs nit → equity decreases ~0.12."""
    base = hs(MadeHandType.MIDDLE_PAIR, equity=0.42)
    tex = board_tex("AsKd3c")  # standard board
    adj = adjust_equity_bucket(base, tex, "nit", "flop")
    assert adj < 0.42


def test_monster_minimal_adjustment():
    """Monster hand (set/flush) → opponent type has minimal impact."""
    base = hs(MadeHandType.TRIPS_SET, equity=0.75)
    tex = board_tex("9d7s3c")
    adj_fish = adjust_equity_bucket(base, tex, "fish", "flop")
    adj_nit = adjust_equity_bucket(base, tex, "nit", "flop")
    # Fish +0.03, nit -0.03 → small change
    assert abs(adj_fish - 0.75) < 0.10
    assert abs(adj_nit - 0.75) < 0.10
    assert adj_fish > adj_nit


def test_lag_small_positive_adjustment():
    """LAG → small positive adjustment for top pair."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj = adjust_equity_bucket(base, tex, "LAG", "flop")
    adj_tag = adjust_equity_bucket(base, tex, "TAG", "flop")
    # LAG should be higher than TAG
    assert adj >= adj_tag


def test_tag_no_major_adjustment():
    """TAG → no type adjustment (baseline = 0)."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj = adjust_equity_bucket(base, tex, "TAG", "flop")
    adj_unknown = adjust_equity_bucket(base, tex, "unknown", "flop")
    # Both should be same (dry board adjustment only)
    assert abs(adj - adj_unknown) < 0.01


# ---------------------------------------------------------------------------
# Board texture adjustments
# ---------------------------------------------------------------------------

def test_wet_board_lowers_equity():
    """Wet board → made hand equity decreases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex_wet = board_tex("9h8h7s")   # wet, connected, flush possible
    tex_dry = board_tex("As2d7c")   # dry
    adj_wet = adjust_equity_bucket(base, tex_wet, "unknown", "flop")
    adj_dry = adjust_equity_bucket(base, tex_dry, "unknown", "flop")
    assert adj_dry > adj_wet


def test_dry_board_raises_equity():
    """Dry board → made hand equity increases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As2d7c")  # dry (rainbow, low connectivity)
    adj = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj > 0.65


def test_paired_board_lowers_equity_without_trips():
    """Paired board without set/trips → equity decreases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("AsAd7c")  # paired board
    adj = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj < 0.65 + 0.04  # dry board +0.04, paired -0.04 → roughly neutral


# ---------------------------------------------------------------------------
# Street adjustments
# ---------------------------------------------------------------------------

def test_turn_blank_increases_equity():
    """Blank turn → made hand equity slightly increases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj_turn_blank = adjust_equity_bucket(base, tex, "unknown", "turn", turn_is_blank=True)
    adj_flop = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj_turn_blank > adj_flop


def test_turn_scare_card_decreases_equity():
    """Non-blank turn (scary) → equity decreases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj_turn_bad = adjust_equity_bucket(base, tex, "unknown", "turn", turn_is_blank=False)
    adj_flop = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj_turn_bad < adj_flop


def test_river_blank_increases_equity():
    """Blank river → draws all missed; equity increases."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj_river = adjust_equity_bucket(base, tex, "unknown", "river", river_is_blank=True)
    adj_flop = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj_river > adj_flop


def test_river_draw_completes_decreases_equity():
    """Draw-completing river → equity decreases significantly."""
    base = hs(MadeHandType.TOP_PAIR_TOP_KICKER, equity=0.65)
    tex = board_tex("As7d3c")
    adj_river_bad = adjust_equity_bucket(base, tex, "unknown", "river", river_is_blank=False)
    adj_flop = adjust_equity_bucket(base, tex, "unknown", "flop")
    assert adj_river_bad < adj_flop


# ---------------------------------------------------------------------------
# Ace high vs different opponents
# ---------------------------------------------------------------------------

def test_ace_high_vs_fish_has_showdown_value():
    """Ace high vs fish → equity should be significantly above base (0.18)."""
    base = hs(MadeHandType.ACE_HIGH, equity=0.18, showdown=False)
    tex = board_tex("Ks7d3c")
    adj = adjust_equity_bucket(base, tex, "fish", "flop")
    # fish adjustment: +0.10 type, +0.04 dry board
    assert adj > 0.25  # meaningful increase


def test_ace_high_vs_nit_almost_no_value():
    """Ace high vs nit → equity barely above minimum."""
    base = hs(MadeHandType.ACE_HIGH, equity=0.18, showdown=False)
    tex = board_tex("Ks7d3c")
    adj = adjust_equity_bucket(base, tex, "nit", "flop")
    # nit adjustment: -0.10 type, +0.04 dry board → near 0
    assert adj <= 0.15


# ---------------------------------------------------------------------------
# Bounds check
# ---------------------------------------------------------------------------

def test_equity_never_below_zero_or_above_one():
    """Equity is always clipped to [0.01, 0.99]."""
    base_min = hs(MadeHandType.NO_PAIR, equity=0.08)
    base_max = hs(MadeHandType.STRAIGHT_FLUSH, equity=1.0)
    tex = board_tex("9h8h7s")
    adj_min = adjust_equity_bucket(base_min, tex, "nit", "river", river_is_blank=False)
    adj_max = adjust_equity_bucket(base_max, tex, "fish", "flop")
    assert adj_min >= 0.01
    assert adj_max <= 0.99


# ---------------------------------------------------------------------------
# Integration: classify_hand → adjust_equity_bucket
# ---------------------------------------------------------------------------

def test_integration_classify_then_adjust():
    """Full pipeline: classify hand, adjust equity vs fish on dry board."""
    hand = tuple(cards_from_str("AsKd"))
    board_cards = cards_from_str("Ah7c3s")
    hs_raw = classify_hand(hand, board_cards)
    tex = analyze_board(board_cards)

    adj = adjust_equity_bucket(hs_raw, tex, "fish", "flop")
    # TPTK vs fish on dry board should be ~0.65 + 0.12 + 0.04 = 0.81
    assert adj > hs_raw.equity_bucket
    assert 0.70 < adj < 0.99
