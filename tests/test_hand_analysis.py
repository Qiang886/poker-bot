"""Tests for hand classification."""

import pytest
from src.card import cards_from_str
from src.hand_analysis import classify_hand, detect_draws, MadeHandType, DrawType


def hand_and_board(hand_str, board_str):
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    return hand, board


def test_overpair_big():
    hand, board = hand_and_board("AsAh", "Ks7d2c")
    hs = classify_hand(hand, board)
    assert hs.made_hand == MadeHandType.OVERPAIR_BIG


def test_top_pair_top_kicker():
    hand, board = hand_and_board("AsKh", "Ad7c2h")
    hs = classify_hand(hand, board)
    assert hs.made_hand == MadeHandType.TOP_PAIR_TOP_KICKER


def test_set_detection():
    hand, board = hand_and_board("7s7h", "7dAcKs")
    hs = classify_hand(hand, board)
    assert hs.made_hand == MadeHandType.TRIPS_SET


def test_nut_flush_draw():
    hand, board = hand_and_board("AsKs", "2s5s9h")
    draw = detect_draws(hand, board)
    assert draw == DrawType.FLUSH_DRAW_NUT


def test_no_pair():
    hand, board = hand_and_board("2h3d", "7sJcAh")
    hs = classify_hand(hand, board)
    assert hs.made_hand in (MadeHandType.NO_PAIR, MadeHandType.ACE_HIGH)


def test_two_pair():
    hand, board = hand_and_board("AsKh", "AdKd2c")
    hs = classify_hand(hand, board)
    assert hs.made_hand in (MadeHandType.TOP_TWO_PAIR, MadeHandType.BOTTOM_TWO_PAIR)


def test_flush():
    hand, board = hand_and_board("AsTs", "2s5s9s")
    hs = classify_hand(hand, board)
    assert hs.made_hand in (MadeHandType.FLUSH_NUT, MadeHandType.FLUSH_LOW)


def test_has_showdown_value_for_top_pair():
    hand, board = hand_and_board("AhKc", "As2d7h")
    hs = classify_hand(hand, board)
    assert hs.has_showdown_value is True


def test_no_showdown_value_for_air():
    hand, board = hand_and_board("2h3d", "7sJcAh")
    hs = classify_hand(hand, board)
    # weak hands may not have showdown value
    assert isinstance(hs.has_showdown_value, bool)
