"""Tests for equity calculator."""

import pytest
from src.card import cards_from_str
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range
from src.position import Position
from src.equity import calculate_equity


def _random_range(dead):
    all_hands = get_rfi_range(Position.BTN)
    return build_range_combos(all_hands, dead)


def test_aa_equity_vs_random_range():
    hero = tuple(cards_from_str("AsAh"))
    dead = list(hero)
    villain_range = _random_range(dead)
    eq = calculate_equity(hero, villain_range, [], num_simulations=500)
    assert eq > 0.80, f"AA equity should be > 80%, got {eq:.2%}"


def test_72o_equity_vs_random_range():
    hero = tuple(cards_from_str("7h2d"))
    dead = list(hero)
    villain_range = _random_range(dead)
    eq = calculate_equity(hero, villain_range, [], num_simulations=500)
    assert eq < 0.40, f"72o equity should be < 40%, got {eq:.2%}"


def test_equity_between_0_and_1():
    hero = tuple(cards_from_str("KsQh"))
    dead = list(hero)
    villain_range = _random_range(dead)
    eq = calculate_equity(hero, villain_range, [], num_simulations=200)
    assert 0.0 <= eq <= 1.0


def test_board_equity():
    hero = tuple(cards_from_str("AsAh"))
    board = cards_from_str("Ad2c7h")
    dead = list(hero) + board
    villain_range = _random_range(dead)
    eq = calculate_equity(hero, villain_range, board, num_simulations=300)
    assert eq > 0.85, f"Set of aces equity should be > 85%, got {eq:.2%}"


def test_empty_range_returns_half():
    hero = tuple(cards_from_str("AsAh"))
    from src.equity import calculate_equity
    eq = calculate_equity(hero, [], [], num_simulations=100)
    assert eq == 0.5
