"""Tests for postflop engine."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range


def make_engine():
    return PostflopEngine()


def decide(hand_str, board_str, pot=10.0, to_call=0.0, stack=100.0, position=Position.BTN):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
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
        street="flop",
        villain_profile=None,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=False,
    )


def test_nut_flush_bets():
    d = decide("AsTs", "2s5s9s")
    assert d.action in ("bet", "raise", "all-in")


def test_tptk_bets_dry_board():
    d = decide("AhKc", "As2d7h")
    assert d.action in ("bet", "raise")


def test_air_folds_to_bet():
    d = decide("2h3d", "AsKhQc", to_call=7.0)
    assert d.action == "fold"


def test_check_with_weak_hand():
    d = decide("2h3d", "AsKhQc", to_call=0.0)
    assert d.action == "check"
