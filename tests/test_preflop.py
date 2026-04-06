"""Tests for preflop engine."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.preflop import PreflopEngine
from src.ranges import get_rfi_range
from src.weighted_range import count_combos


def make_engine():
    return PreflopEngine()


def decide(hand_str, position, action_history=None):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    return engine.decide(
        hero_hand=hand,
        position=position,
        action_history=action_history or [],
        pot=1.5,
        to_call=0.0,
        hero_stack=100.0,
        num_players=6,
        villain_profiles={},
    )


def test_aa_utg_raises():
    d = decide("AsAh", Position.UTG)
    assert d.action == "raise"


def test_aa_btn_raises():
    d = decide("AsAh", Position.BTN)
    assert d.action == "raise"


def test_72o_utg_folds():
    d = decide("7h2d", Position.UTG)
    assert d.action == "fold"


def test_aks_btn_raises():
    d = decide("AsKs", Position.BTN)
    assert d.action == "raise"


def test_kk_facing_raise_3bets():
    engine = make_engine()
    hand = tuple(cards_from_str("KsKh"))
    action_history = [{"position": Position.UTG, "action": "raise", "amount": 2.5}]
    d = engine.decide(
        hero_hand=hand,
        position=Position.BTN,
        action_history=action_history,
        pot=4.0,
        to_call=2.5,
        hero_stack=100.0,
        num_players=6,
        villain_profiles={},
    )
    assert d.action in ("3bet", "raise")


def test_btn_wider_rfi_than_utg():
    utg_combos = count_combos(get_rfi_range(Position.UTG))
    btn_combos = count_combos(get_rfi_range(Position.BTN))
    assert btn_combos > utg_combos
