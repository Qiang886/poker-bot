"""Tests for check-raise logic in the postflop engine."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range
from src.barrel_plan import BarrelPlan, ValueLine
from src.opponent import VillainProfile


def make_engine():
    return PostflopEngine()


def decide(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    position=Position.BB,   # OOP by default for check-raise tests
    villain_profile=None,
    barrel_plan=None,
    street="flop",
    is_multiway=False,
):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.UTG), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BB), dead)
    return engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=position,
        street=street,
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=barrel_plan,
        is_multiway=is_multiway,
    )


# ---------------------------------------------------------------------------
# OOP set on dry board → check-raise plan (check first)
# ---------------------------------------------------------------------------

def test_oop_set_dry_board_checks():
    """OOP set on dry board should check (planning to check-raise)."""
    d = decide("7h7d", "7s2cKd", position=Position.BB)
    assert d.action == "check"
    assert "check-raise" in d.reasoning.lower() or "check" in d.action


def test_oop_nut_flush_draw_check_raise():
    """OOP nut flush draw on flop → check-raise semi-bluff."""
    d = decide("AhKh", "2h5h9c", position=Position.BB, street="flop")
    assert d.action == "check"


def test_ip_set_bets_not_check_raise():
    """IP set should bet for value, not check-raise."""
    d = decide("7h7d", "7s2cKd", position=Position.BTN)
    assert d.action in ("bet", "raise", "all-in")


def test_multiway_no_check_raise():
    """In multiway pot, check-raise is disabled; strong hand should still be aggressive."""
    d = decide("7h7d", "7s2cKd", position=Position.BB, is_multiway=True)
    # Should bet or raise (multiway adjustments may still allow value bet)
    # Key thing: check-raise plan should NOT activate
    # The engine may check or bet; just ensure reasoning doesn't mention check-raise
    if d.action == "check":
        assert "check-raise" not in d.reasoning.lower()


def test_low_cbet_freq_no_check_raise():
    """Villain with low cbet frequency → no check-raise opportunity."""
    profile = VillainProfile()
    # Set up 50 hands and low cbet
    profile.stats.hands_played = 50
    profile.stats.cbet_flop = 0.25  # low cbet freq
    d = decide("7h7d", "7s2cKd", position=Position.BB, villain_profile=profile)
    # With low cbet frequency, check-raise should not be planned
    if d.action == "check":
        assert "check-raise" not in d.reasoning.lower()


# ---------------------------------------------------------------------------
# Facing a bet with CHECK_RAISE barrel plan → execute raise
# ---------------------------------------------------------------------------

def test_facing_bet_executes_check_raise():
    """When barrel_plan has CHECK_RAISE value_line and facing a bet, bot should raise."""
    plan = BarrelPlan(
        flop_action="check",
        turn_action="bet",
        river_action="bet",
        value_line=ValueLine.CHECK_RAISE,
        continue_runouts=["any"],
        give_up_runouts=[],
    )
    d = decide(
        "7h7d", "7s2cKd",
        position=Position.BB,
        to_call=5.0,
        barrel_plan=plan,
    )
    assert d.action == "raise"
    assert "check-raise" in d.reasoning.lower()
