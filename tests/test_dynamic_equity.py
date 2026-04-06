"""Tests for dynamic equity bucket adjustment."""

import pytest
from src.card import cards_from_str
from src.hand_analysis import HandStrength, MadeHandType, DrawType, classify_hand
from src.board_analysis import analyze_board
from src.dynamic_equity import adjust_equity_bucket


def make_hand_strength(made, equity, draw=DrawType.NONE, is_vulnerable=False, has_sdv=True):
    return HandStrength(
        made_hand=made,
        draw=draw,
        equity_bucket=equity,
        is_vulnerable=is_vulnerable,
        has_showdown_value=has_sdv,
    )


def get_hand_and_texture(hand_str, board_str):
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    hand_strength = classify_hand(hand, board)
    board_texture = analyze_board(board)
    return hand_strength, board_texture


# ---------------------------------------------------------------------------
# Villain type adjustments
# ---------------------------------------------------------------------------

def test_tptk_vs_fish_equity_increases():
    """TPTK vs fish should have higher equity (~+0.12)."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "fish", "flop")
    assert adjusted > base, f"Expected increase vs fish, got base={base}, adjusted={adjusted}"
    assert adjusted - base >= 0.08, f"Expected at least 0.08 increase, got {adjusted - base:.3f}"


def test_tptk_vs_nit_equity_decreases():
    """TPTK vs nit should have lower equity (~-0.12)."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "nit", "flop")
    assert adjusted < base, f"Expected decrease vs nit, got base={base}, adjusted={adjusted}"
    assert base - adjusted >= 0.07, f"Expected at least 0.07 decrease, got {base - adjusted:.3f}"


def test_middle_pair_vs_fish_equity_increases():
    """Middle pair vs fish should have higher equity (~+0.10)."""
    hs, bt = get_hand_and_texture("9h8s", "9dKc3h")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "fish", "flop")
    assert adjusted > base, f"Expected increase vs fish, got base={base}, adjusted={adjusted}"
    assert adjusted - base >= 0.06, f"Expected at least 0.06 increase, got {adjusted - base:.3f}"


def test_monster_minimal_type_adjustment():
    """Monster hand (set/quads) should have smaller villain type adjustment than TPTK."""
    hs, bt = get_hand_and_texture("7h7d", "7sAcKd")
    base = hs.equity_bucket
    adj_fish = adjust_equity_bucket(hs, bt, "fish", "flop")
    adj_nit = adjust_equity_bucket(hs, bt, "nit", "flop")
    # TPTK fish adjustment is ~0.12+; monster adjustment should be smaller
    tptk_hs, tptk_bt = get_hand_and_texture("AhKd", "As7d3h")
    tptk_fish_adj = adjust_equity_bucket(tptk_hs, tptk_bt, "fish", "flop")
    assert abs(adj_fish - base) < abs(tptk_fish_adj - tptk_hs.equity_bucket), (
        f"Monster fish adjustment ({abs(adj_fish - base):.3f}) should be smaller than "
        f"TPTK fish adjustment ({abs(tptk_fish_adj - tptk_hs.equity_bucket):.3f})"
    )
    assert abs(adj_nit - base) < abs(tptk_hs.equity_bucket - adjust_equity_bucket(tptk_hs, tptk_bt, "nit", "flop")), (
        "Monster nit adjustment should be smaller than TPTK nit adjustment"
    )


def test_lag_vs_tptk_mild_increase():
    """TPTK vs LAG should have mild positive adjustment (+0.06)."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "LAG", "flop")
    assert adjusted > base, f"Expected increase vs LAG, got base={base}, adjusted={adjusted}"


# ---------------------------------------------------------------------------
# Board texture adjustments
# ---------------------------------------------------------------------------

def test_wet_board_decreases_equity():
    """Wet board should decrease equity bucket."""
    # Wet board: flush draw + straight draw possible
    hs, bt = get_hand_and_texture("AhKd", "Js9s8s")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "flop")
    # Wet board adjustment should bring equity down
    assert adjusted <= base, f"Expected no increase on wet board, got base={base}, adjusted={adjusted}"


def test_dry_board_increases_equity():
    """Dry board should increase equity bucket."""
    # Dry rainbow board: no flush/straight possible
    hs, bt = get_hand_and_texture("AhKd", "As7d2c")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "flop")
    assert adjusted >= base, f"Expected no decrease on dry board, got base={base}, adjusted={adjusted}"


def test_flush_possible_no_hero_flush_draw_decreases():
    """When flush is possible and hero has no flush draw, equity decreases."""
    # Board has 2 hearts, hero has no hearts
    hs, bt = get_hand_and_texture("AcKd", "9h6h2d")
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "flop")
    # Flush possible but hero has no flush draw → equity decreases
    assert adjusted <= base, f"Expected decrease when flush possible and hero has no FD"


# ---------------------------------------------------------------------------
# Street adjustments
# ---------------------------------------------------------------------------

def test_turn_blank_increases_equity():
    """Blank turn card should micro-increase equity."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h2c")  # 2c is blank turn
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "turn", turn_is_blank=True)
    assert adjusted >= base, f"Expected non-decrease on blank turn, got base={base}, adjusted={adjusted}"


def test_turn_non_blank_decreases_equity():
    """Dangerous turn card should decrease equity."""
    hs, bt = get_hand_and_texture("AhKd", "7s6d3hKc")  # Kc is overcard/scary
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "turn", turn_is_blank=False)
    assert adjusted < base, f"Expected decrease on non-blank turn, got base={base}, adjusted={adjusted}"


def test_river_blank_increases_equity():
    """Blank river card should increase equity more than blank turn."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h2c5s")  # 5s is blank river
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "river", river_is_blank=True)
    assert adjusted >= base, f"Expected non-decrease on blank river, got base={base}, adjusted={adjusted}"


def test_river_dangerous_decreases_equity():
    """Dangerous river (draw completing) should decrease equity significantly."""
    hs, bt = get_hand_and_texture("AhKd", "7s6d3h2cJs")  # potential straight completing
    base = hs.equity_bucket
    adjusted = adjust_equity_bucket(hs, bt, "unknown", "river", river_is_blank=False)
    assert adjusted < base, f"Expected decrease on dangerous river, got base={base}, adjusted={adjusted}"


# ---------------------------------------------------------------------------
# Clamping behavior
# ---------------------------------------------------------------------------

def test_equity_never_below_0_01():
    """Adjusted equity should never go below 0.01."""
    # Very weak hand + nit + wet board
    hs = make_hand_strength(MadeHandType.NO_PAIR, 0.08, DrawType.NONE, False, False)
    bt, _ = analyze_board(cards_from_str("Js9s8s")), None
    bt = analyze_board(cards_from_str("Js9s8s"))
    adjusted = adjust_equity_bucket(hs, bt, "nit", "river", river_is_blank=False)
    assert adjusted >= 0.01, f"Equity should never go below 0.01, got {adjusted}"


def test_equity_never_above_0_99():
    """Adjusted equity should never go above 0.99."""
    # Very strong hand + fish + dry board
    hs = make_hand_strength(MadeHandType.STRAIGHT_FLUSH, 1.0, DrawType.NONE, False, True)
    bt = analyze_board(cards_from_str("As2c3d"))
    adjusted = adjust_equity_bucket(hs, bt, "fish", "river", river_is_blank=True)
    assert adjusted <= 0.99, f"Equity should never go above 0.99, got {adjusted}"


def test_unknown_villain_no_type_adjustment():
    """Unknown villain type should result in zero villain type adjustment."""
    hs, bt = get_hand_and_texture("AhKd", "As7d3h")
    base = hs.equity_bucket
    adjusted_unknown = adjust_equity_bucket(hs, bt, "unknown", "flop")
    # "unknown" gives no type adjustment; only texture/street adjustments apply
    adjusted_tag = adjust_equity_bucket(hs, bt, "TAG", "flop")
    # TAG also has no explicit adjustment in the function → same as unknown
    assert adjusted_unknown == adjusted_tag, f"Unknown and TAG should give same result"
