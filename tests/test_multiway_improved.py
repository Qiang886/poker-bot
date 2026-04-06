"""Tests for improved multiway pot strategy."""

import pytest
from src.card import cards_from_str
from src.hand_analysis import classify_hand, MadeHandType, DrawType, HandStrength
from src.board_analysis import analyze_board
from src.multiway import adjust_for_multiway, MultiwayAdjustment


def make_hand_strength(made: MadeHandType, draw: DrawType = DrawType.NONE) -> HandStrength:
    return HandStrength(
        made_hand=made,
        draw=draw,
        equity_bucket=0.5,
        is_vulnerable=False,
        has_showdown_value=(made >= MadeHandType.BOTTOM_PAIR),
    )


def make_texture(board_str):
    return analyze_board(cards_from_str(board_str))


# ---------------------------------------------------------------------------
# Heads-up: no adjustments
# ---------------------------------------------------------------------------

def test_headsup_no_adjustment():
    """Heads-up pot should have no adjustments."""
    hs = make_hand_strength(MadeHandType.MIDDLE_PAIR)
    texture = make_texture("AsTc2d")
    adj = adjust_for_multiway(hs, num_players=2, board_texture=texture)
    assert adj.bet_frequency_multiplier == 1.0
    assert adj.value_threshold_adjustment == 0
    assert adj.bluff_allowed is True
    assert adj.sizing_multiplier == 1.0


# ---------------------------------------------------------------------------
# 3-way pot tests
# ---------------------------------------------------------------------------

def test_3way_reduced_bet_frequency():
    """3-way pot should significantly reduce bet frequency."""
    hs = make_hand_strength(MadeHandType.MIDDLE_PAIR)
    texture = make_texture("AsTc2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    assert adj.bet_frequency_multiplier < 1.0
    assert adj.bet_frequency_multiplier >= 0.45  # not too tight


def test_3way_combo_draw_bluff_allowed():
    """3-way pot: combo draw should be allowed to semi-bluff."""
    hs = make_hand_strength(MadeHandType.NO_PAIR, DrawType.COMBO_DRAW)
    texture = make_texture("9h8h2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    assert adj.bluff_allowed is True


def test_3way_single_draw_bluff_not_allowed():
    """3-way pot: single draw (gutshot) should not bluff."""
    hs = make_hand_strength(MadeHandType.NO_PAIR, DrawType.GUTSHOT)
    texture = make_texture("9h8c2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    assert adj.bluff_allowed is False


def test_3way_sizing_multiplier_increased():
    """3-way pot sizing should be larger than heads-up."""
    hs = make_hand_strength(MadeHandType.TOP_PAIR_TOP_KICKER)
    texture = make_texture("AsTc2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    assert adj.sizing_multiplier > 1.0


def test_3way_value_threshold_tighter():
    """3-way pot requires stronger hands for value betting."""
    hs = make_hand_strength(MadeHandType.MIDDLE_PAIR)
    texture = make_texture("AsTc2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    # Threshold should require at least TPTK
    assert adj.value_threshold_adjustment > 0


# ---------------------------------------------------------------------------
# 4+ way pot tests
# ---------------------------------------------------------------------------

def test_4way_minimal_bluff():
    """4-way pot: only nut combo draws can bluff."""
    hs_nut = make_hand_strength(MadeHandType.NO_PAIR, DrawType.COMBO_DRAW_NUT)
    hs_flush = make_hand_strength(MadeHandType.NO_PAIR, DrawType.FLUSH_DRAW_NUT)
    hs_oesd = make_hand_strength(MadeHandType.NO_PAIR, DrawType.OESD)
    texture = make_texture("9h8h2d")
    adj_nut = adjust_for_multiway(hs_nut, num_players=4, board_texture=texture)
    adj_flush = adjust_for_multiway(hs_flush, num_players=4, board_texture=texture)
    adj_oesd = adjust_for_multiway(hs_oesd, num_players=4, board_texture=texture)
    assert adj_nut.bluff_allowed is True    # nut combo draw allowed
    assert adj_flush.bluff_allowed is False  # regular flush draw not allowed
    assert adj_oesd.bluff_allowed is False   # OESD not allowed


def test_4way_very_low_bet_frequency():
    """4-way pot should have very low bet frequency."""
    hs = make_hand_strength(MadeHandType.TOP_PAIR_TOP_KICKER)
    texture = make_texture("AsTc2d")
    adj = adjust_for_multiway(hs, num_players=4, board_texture=texture)
    assert adj.bet_frequency_multiplier <= 0.40


def test_4way_higher_sizing_than_3way():
    """4-way pot sizing should be higher than 3-way."""
    hs = make_hand_strength(MadeHandType.FULL_HOUSE)
    texture = make_texture("AsTc2d")
    adj3 = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    adj4 = adjust_for_multiway(hs, num_players=4, board_texture=texture)
    assert adj4.sizing_multiplier >= adj3.sizing_multiplier


def test_wet_board_further_tightens():
    """Wet board (wetness >= 7) should reduce bet frequency further."""
    hs = make_hand_strength(MadeHandType.TOP_PAIR_GOOD_KICKER)
    dry_texture = make_texture("As3c2d")
    wet_texture = make_texture("9h8h7h")
    adj_dry = adjust_for_multiway(hs, num_players=3, board_texture=dry_texture)
    adj_wet = adjust_for_multiway(hs, num_players=3, board_texture=wet_texture)
    # Wet board should have equal or lower bet frequency
    assert adj_wet.bet_frequency_multiplier <= adj_dry.bet_frequency_multiplier


def test_draw_continue_threshold_3way():
    """3-way pot draw threshold should require combo draw or better."""
    hs = make_hand_strength(MadeHandType.NO_PAIR, DrawType.FLUSH_DRAW_LOW)
    texture = make_texture("9h8h2d")
    adj = adjust_for_multiway(hs, num_players=3, board_texture=texture)
    # Should at least require COMBO_DRAW to continue (not just gutshot/flush_draw_low)
    assert adj.draw_continue_threshold >= DrawType.COMBO_DRAW


def test_draw_continue_threshold_4way():
    """4-way pot draw threshold should require flush_draw_nut or better."""
    hs = make_hand_strength(MadeHandType.NO_PAIR, DrawType.OESD)
    texture = make_texture("9h8h2d")
    adj = adjust_for_multiway(hs, num_players=4, board_texture=texture)
    assert adj.draw_continue_threshold >= DrawType.FLUSH_DRAW_NUT
