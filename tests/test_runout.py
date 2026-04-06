"""Tests for runout analysis and get_current_action runout awareness."""

import pytest
from src.card import Card, Rank, Suit, card_from_str, cards_from_str
from src.barrel_plan import (
    analyze_runout,
    get_current_action,
    BarrelPlan,
    ValueLine,
    BluffLine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def c(s: str) -> Card:
    return card_from_str(s)


def board(s: str):
    return cards_from_str(s)


# ---------------------------------------------------------------------------
# analyze_runout tag tests
# ---------------------------------------------------------------------------

def test_flush_completes():
    """K♥9♥4♣ + 7♥ → flush_completes (3 hearts now)."""
    flop = board("Kh9h4c")
    turn = c("7h")
    tags = analyze_runout(flop, turn)
    assert "flush_completes" in tags


def test_overcard_and_scary_card():
    """9♦7♠3♣ + A♥ → overcard and scary_card."""
    flop = board("9d7s3c")
    turn = c("Ah")
    tags = analyze_runout(flop, turn)
    assert "overcard" in tags
    assert "scary_card" in tags


def test_paired_board():
    """K♠Q♥7♦ + 7♣ → paired_board."""
    flop = board("KsQh7d")
    turn = c("7c")
    tags = analyze_runout(flop, turn)
    assert "paired_board" in tags


def test_blank_low_card():
    """A♠K♦9♣ + 3♥ → blank and low_card."""
    flop = board("AsKd9c")
    turn = c("3h")
    tags = analyze_runout(flop, turn)
    assert "blank" in tags
    assert "low_card" in tags


def test_straight_completes():
    """J♠9♥8♦ + T♣ → straight_completes (J-T-9-8 connected)."""
    flop = board("Js9h8d")
    turn = c("Tc")
    tags = analyze_runout(flop, turn)
    assert "straight_completes" in tags


def test_no_blank_when_danger():
    """Flush completing card should NOT be tagged as blank."""
    flop = board("Kh9h4c")
    turn = c("7h")
    tags = analyze_runout(flop, turn)
    assert "blank" not in tags


def test_king_scary_on_low_board():
    """7♣5♦2♠ + K♥ → scary_card."""
    flop = board("7c5d2s")
    turn = c("Kh")
    tags = analyze_runout(flop, turn)
    assert "scary_card" in tags


def test_monotone_4():
    """K♥9♥4♥ + 7♥ → monotone_4 (4 hearts)."""
    flop = board("Kh9h4h")
    turn = c("7h")
    tags = analyze_runout(flop, turn)
    assert "monotone_4" in tags


def test_river_paired_board():
    """A♠K♦9♣7♥ + 9♦ → paired_board."""
    turn_board = board("AsKd9c7h")
    river = c("9d")
    tags = analyze_runout(turn_board, river)
    assert "paired_board" in tags


# ---------------------------------------------------------------------------
# get_current_action runout-aware tests
# ---------------------------------------------------------------------------

def _make_plan(give_up=None, continue_r=None, turn_action="bet", river_action="bet"):
    return BarrelPlan(
        flop_action="bet",
        turn_action=turn_action,
        river_action=river_action,
        value_line=ValueLine.BET_BET_BET,
        give_up_runouts=give_up or [],
        continue_runouts=continue_r or ["any"],
    )


def test_give_up_on_flush_completes_turn():
    """Bot should check on turn when flush completes and give_up_runouts includes flush_completes."""
    plan = _make_plan(give_up=["flush_completes"])
    # board = flop + turn; turn is index 3
    full_board = board("Kh9h4c7h")  # 7h completes flush
    action = get_current_action(plan, "turn", turn_card=full_board[3], board=full_board)
    assert action == "check"


def test_continue_on_blank_turn():
    """Bot should bet on blank turn when plan says continue on blank."""
    plan = _make_plan(give_up=["flush_completes"], continue_r=["blank"])
    full_board = board("AsKd9c3h")  # 3h = blank
    action = get_current_action(plan, "turn", turn_card=full_board[3], board=full_board)
    assert action == "bet"


def test_give_up_on_scary_card_river():
    """Bluff plan should check on river when scary_card appears."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bluff",
        bluff_line=BluffLine.BLUFF_DOUBLE_BARREL,
        give_up_runouts=[],
        continue_runouts=["any"],
    )
    full_board = board("9d7s3c2hAh")  # Ah = scary_card
    action = get_current_action(plan, "river", river_card=full_board[4], board=full_board)
    assert action == "check"


def test_continue_bluff_on_blank_river():
    """Bluff plan should continue on blank river."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bluff",
        bluff_line=BluffLine.BLUFF_DOUBLE_BARREL,
        give_up_runouts=[],
        continue_runouts=["any"],
    )
    full_board = board("AsKd9c7h3c")  # 3c = blank
    action = get_current_action(plan, "river", river_card=full_board[4], board=full_board)
    assert action == "bluff"


def test_no_board_falls_back_to_plan():
    """Without board info, fall back to plan turn_action."""
    plan = _make_plan(give_up=["flush_completes"])
    action = get_current_action(plan, "turn")
    assert action == "bet"


def test_give_up_any_still_respected():
    """give_up_runouts=['any'] should still trigger check without board."""
    plan = _make_plan(give_up=["any"])
    action = get_current_action(plan, "turn")
    assert action == "check"
